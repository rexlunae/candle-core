//! Vulkan compute kernels for the [`super`] backend.
//!
//! GLSL kernel source is embedded here and compiled to SPIR-V at runtime with
//! `naga` (pure Rust), so no external `glslc`/`glslangValidator` is needed to
//! build or run. Each kernel argument that is a tensor lives in a
//! STORAGE_BUFFER SSBO; scalar dimensions (n / m·n·k) go in a small
//! uniform-aligned push range.
//!
//! Native execution is gated behind `JOSHUA_VULKAN_NATIVE=1` (mirroring
//! `JOSHUA_OPENCL_NATIVE`), so the default behaviour of every op is the CPU
//! fallback and these kernels only run when explicitly enabled on Vulkan
//! hardware.

use crate::{Error, Result};
use super::{VulkanDevice, VulkanStorage};
use ash::vk;

// ---------------------------------------------------------------------------
// Native-kernel gating (mirrors the OpenCL backend's opt-in).
// ---------------------------------------------------------------------------
static NATIVE_EXEC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn env_flag() -> bool {
    std::env::var("JOSHUA_VULKAN_NATIVE").map(|v| v == "1").unwrap_or(false)
}

/// Whether native Vulkan kernels are enabled (`JOSHUA_VULKAN_NATIVE=1`).
pub fn native_enabled() -> bool {
    env_flag()
}

/// Number of native kernel dispatches so far (diagnostic counter).
pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn note_native_exec() {
    NATIVE_EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Log why a native Vulkan kernel fell back to CPU, when
/// `JOSHUA_VULKAN_DEBUG=1`. Silent fallback hides device loss / misconfiguration,
/// so this gives production runs an opt-in diagnostic.
pub(crate) fn log_native_fallback(op: &str, err: &Error) {
    if std::env::var("JOSHUA_VULKAN_DEBUG").map(|v| v == "1").unwrap_or(false) {
        eprintln!("vulkan: {op} fell back to CPU: {err}");
    }
}

/// Which unary GLSL ops we can dispatch (name -> GLSL expression).
pub fn has_unary(name: &'static str) -> bool {
    matches!(
        name,
        "exp" | "sin" | "cos" | "tan" | "sqrt" | "abs" | "neg" | "ln" | "sqr"
    )
}

/// Which binary GLSL ops we can dispatch (name -> GLSL expression).
pub fn has_binary(name: &'static str) -> bool {
    matches!(name, "add" | "mul" | "sub" | "div" | "pow")
}

// ---------------------------------------------------------------------------
// GLSL kernels (compute).
// ---------------------------------------------------------------------------

/// Elementwise: out[i] = f(buf[i]).
const GLSL_UNARY: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InBuf { float inp[]; };
layout(set = 0, binding = 1) buffer OutBuf { float outp[]; };
layout(push_constant) uniform PC { uint N; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) return;
    float x = inp[i];
    float r = {EXPR};
    outp[i] = r;
}
"#;

/// Affine: out[i] = a * buf[i] + b.
const GLSL_AFFINE: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InBuf { float inp[]; };
layout(set = 0, binding = 1) buffer OutBuf { float outp[]; };
layout(push_constant) uniform PC { uint N; float a; float b; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) return;
    outp[i] = pc.a * inp[i] + pc.b;
}
"#;

/// Binary: out[i] = f(lhs[i], rhs[i]).
const GLSL_BINARY: &str = r#"
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) readonly buffer InBuf0 { float lhs[]; };
layout(set = 0, binding = 1) readonly buffer InBuf1 { float rhs[]; };
layout(set = 0, binding = 2) buffer OutBuf { float outp[]; };
layout(push_constant) uniform PC { uint N; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N) return;
    float a = lhs[i];
    float b = rhs[i];
    outp[i] = {EXPR};
}
"#;

/// Matmul (row-major, single non-batched tile): o[r*N+c] = sum_k a[r*K+k]*b[k*N+c].
///
/// A register-blocked kernel: each thread computes a 4x1 output tile (four rows
/// in one column) directly from global memory, reusing each B element across the
/// four output rows (4x less global traffic vs the naive 1-output kernel). It
/// uses only scalar arithmetic (no shared memory / barriers / vectors), so it is
/// race-free and bit-exact. Boundary reads/writes are guarded so non-multiple-of
/// M/N stay correct. The reduced traffic also stops a large single dispatch
/// (e.g. K=4096) from tripping the iGPU compute watchdog / device-lost.
const GLSL_MATMUL: &str = r#"
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) readonly buffer InBufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer InBufB { float b[]; };
layout(set = 0, binding = 2) buffer OutBuf { float o[]; };
// bsk/bsn are the RHS layout strides along k and n, so a single kernel handles
// a contiguous B ([K,1]), transposed B ([1,K] — weight-transpose) and broadcast
// B ([0,1] — attention KV broadcast). LHS is always row-major. ba/bb/bo are the
// per-batch strides of A/B/o; the batch index rides the workgroup's Z axis.
layout(push_constant) uniform PC {
    uint M; uint N; uint K;
    uint bsk; uint bsn;
    uint ba; uint bb; uint bo;
} pc;
void main() {
    uint tx = gl_LocalInvocationID.x;
    uint ty = gl_LocalInvocationID.y;
    uint bz = gl_WorkGroupID.z;
    uint row0 = gl_WorkGroupID.y * 64u + ty * 4u;
    uint col = gl_WorkGroupID.x * 16u + tx;
    uint ao = bz * pc.ba; // A base offset for this batch
    uint bo_ptr = bz * pc.bb; // B base offset for this batch
    uint oo = bz * pc.bo; // O base offset for this batch
    float c0 = 0.0;
    float c1 = 0.0;
    float c2 = 0.0;
    float c3 = 0.0;
    for (uint k = 0u; k < pc.K; k++) {
        float a0 = (row0 + 0u < pc.M) ? a[ao + (row0 + 0u) * pc.K + k] : 0.0;
        float a1 = (row0 + 1u < pc.M) ? a[ao + (row0 + 1u) * pc.K + k] : 0.0;
        float a2 = (row0 + 2u < pc.M) ? a[ao + (row0 + 2u) * pc.K + k] : 0.0;
        float a3 = (row0 + 3u < pc.M) ? a[ao + (row0 + 3u) * pc.K + k] : 0.0;
        float bv = (col < pc.N) ? b[bo_ptr + k * pc.bsk + col * pc.bsn] : 0.0;
        c0 += a0 * bv;
        c1 += a1 * bv;
        c2 += a2 * bv;
        c3 += a3 * bv;
    }
    if (col < pc.N) {
        if (row0 + 0u < pc.M) o[oo + (row0 + 0u) * pc.N + col] = c0;
        if (row0 + 1u < pc.M) o[oo + (row0 + 1u) * pc.N + col] = c1;
        if (row0 + 2u < pc.M) o[oo + (row0 + 2u) * pc.N + col] = c2;
        if (row0 + 3u < pc.M) o[oo + (row0 + 3u) * pc.N + col] = c3;
    }
}
"#;

/// Reduce the last dimension of a contiguous f32 tensor (one row per thread):
/// out[r] = sum|max|mean over the row's COLS elements. This is the primitive
/// softmax and RMSNorm need (max over the row, then sum over the row).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ReduceLastDimOp {
    Sum,
    Max,
    Mean,
}

const GLSL_REDUCE_LASTDIM: &str = r#"
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer InBuf { float x[]; };
layout(set = 0, binding = 1) buffer OutBuf { float o[]; };
layout(push_constant) uniform PC { uint ROWS; uint COLS; uint OP; } pc;
float acc_of(float a, float b) {
    if (pc.OP == 1u) return max(a, b);      // max
    else return a + b;                       // sum (mean divides at the end)
}
void main() {
    uint r = gl_GlobalInvocationID.x;
    if (r >= pc.ROWS) return;
    uint base = r * pc.COLS;
    // Max initializes from the row's first element (correct IEEE-754 identity
    // for -inf) and skips it in the loop; sum starts at 0.0 and includes it.
    float acc = (pc.OP == 1u) ? x[base] : 0.0;
    uint cs = (pc.OP == 1u) ? 1u : 0u;
    for (uint c = cs; c < pc.COLS; c++) {
        acc = acc_of(acc, x[base + c]);
    }
    if (pc.OP == 2u) acc = acc / float(pc.COLS); // mean
    o[r] = acc;
}
"#;

// ---------------------------------------------------------------------------
// naga GLSL -> SPIR-V.
// ---------------------------------------------------------------------------

fn glsl_to_spirv(source: &str, entry: &str) -> Result<Vec<u32>> {
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use naga::ShaderStage;

    let options = Options::from(ShaderStage::Compute);
    let mut frontend = Frontend::default();
    let module = frontend
        .parse(&options, source)
        .map_err(|e| Error::Msg(format!("vulkan shader parse failed: {e:?}")))?;

    let caps = Capabilities::all();
    let mut validator = Validator::new(ValidationFlags::all(), caps);
    let info = validator
        .validate(&module)
        .map_err(|e| Error::Msg(format!("vulkan shader validation failed: {e:?}")))?;

    let spv_options = spv::Options::default();
    let mut writer = spv::Writer::new(&spv_options)
        .map_err(|e| Error::Msg(format!("vulkan spv writer init failed: {e:?}")))?;
    let mut words: Vec<u32> = Vec::new();
    let pipeline_options = spv::PipelineOptions {
        shader_stage: ShaderStage::Compute,
        entry_point: entry.to_string(),
    };
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .map_err(|e| Error::Msg(format!("vulkan spv write failed: {e:?}")))?;
    Ok(words)
}

// ---------------------------------------------------------------------------
// One dispatch.
// ---------------------------------------------------------------------------

/// A compiled pipeline + its command pool/descriptor set, holding the rare
/// state that doesn't change per call. Rebuilt per dispatch for simplicity
/// (kernel count is tiny in a benchmark); the hot path remains the CPU fallback
/// unless native kernels are explicitly enabled.
struct Dispatch {
    device: ash::Device,
    _pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    _descriptor_set_layout: vk::DescriptorSetLayout,
    _descriptor_pool: vk::DescriptorPool,
    _command_pool: vk::CommandPool,
    _shader_module: vk::ShaderModule,
}

impl Dispatch {
    fn new(dev: &VulkanDevice, spirv: &[u32], buffer_count: u32, push_size: u32) -> Result<Self> {
        let ash_dev = dev.device();

        let sm_ci = vk::ShaderModuleCreateInfo::default().code(spirv);
        let shader_module = match unsafe { ash_dev.create_shader_module(&sm_ci, None) } {
            Ok(m) => m,
            Err(e) => return Err(Error::Msg(format!("vulkan create_shader_module failed: {e:?}"))),
        };
        // On any later failure we must unwind every handle we already created.
        macro_rules! cleanup_and_fail {
            ($fn:expr) => {{
                let e = $fn;
                unsafe {
                    ash_dev.destroy_shader_module(shader_module, None);
                }
                return Err(e);
            }};
        }

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..buffer_count)
            .map(|bi| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(bi)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout =
            match unsafe { ash_dev.create_descriptor_set_layout(&dsl_info, None) } {
                Ok(l) => l,
                Err(e) => cleanup_and_fail!(Error::Msg(format!(
                    "vulkan create_descriptor_set_layout failed: {e:?}"
                ))),
            };

        let push_ranges: Vec<vk::PushConstantRange> = if push_size > 0 {
            vec![vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_size,
            }]
        } else {
            vec![]
        };
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout =
            match unsafe { ash_dev.create_pipeline_layout(&pipeline_layout_info, None) } {
                Ok(l) => l,
                Err(e) => {
                    unsafe {
                        ash_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                        ash_dev.destroy_shader_module(shader_module, None);
                    }
                    return Err(Error::Msg(format!(
                        "vulkan create_pipeline_layout failed: {e:?}"
                    )));
                }
            };

        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry);
        let compute_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline_stack = [compute_info];
        let pipelines = match unsafe {
            ash_dev.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_stack, None)
        } {
            Ok(ps) => ps,
            Err(e) => {
                unsafe {
                    ash_dev.destroy_pipeline_layout(pipeline_layout, None);
                    ash_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    ash_dev.destroy_shader_module(shader_module, None);
                }
                return Err(Error::Msg(format!(
                    "vulkan create_compute_pipelines failed: {:?}",
                    e.1
                )));
            }
        };
        let pipeline = pipelines[0];

        let pool_sizes: Vec<vk::DescriptorPoolSize> = (0..buffer_count)
            .map(|_| vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
            })
            .collect();
        let pool = match unsafe {
            ash_dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        } {
            Ok(p) => p,
            Err(e) => {
                unsafe {
                    ash_dev.destroy_pipeline(pipeline, None);
                    ash_dev.destroy_pipeline_layout(pipeline_layout, None);
                    ash_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    ash_dev.destroy_shader_module(shader_module, None);
                }
                return Err(Error::Msg(format!(
                    "vulkan create_descriptor_pool failed: {e:?}"
                )));
            }
        };

        let cmd_pool = match unsafe {
            ash_dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(dev.queue_family()),
                None,
            )
        } {
            Ok(c) => c,
            Err(e) => {
                unsafe {
                    ash_dev.destroy_descriptor_pool(pool, None);
                    ash_dev.destroy_pipeline(pipeline, None);
                    ash_dev.destroy_pipeline_layout(pipeline_layout, None);
                    ash_dev.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    ash_dev.destroy_shader_module(shader_module, None);
                }
                return Err(Error::Msg(format!(
                    "vulkan create_command_pool failed: {e:?}"
                )));
            }
        };

        Ok(Dispatch {
            device: ash_dev.clone(),
            _pipeline_layout: pipeline_layout,
            pipeline,
            _descriptor_set_layout: descriptor_set_layout,
            _descriptor_pool: pool,
            _command_pool: cmd_pool,
            _shader_module: shader_module,
        })
    }

    fn run(
        &self,
        dev: &VulkanDevice,
        buffers: &[vk::Buffer],
        _local: (u32, u32, u32),
        global: (u32, u32, u32),
        push: &[u8],
    ) -> Result<()> {
        let ash_dev = &self.device;
        // Descriptor set (layout 0).
        let dls = &self._descriptor_set_layout;
        let mut desc_set_handle = vk::DescriptorSet::null();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self._descriptor_pool)
            .set_layouts(std::slice::from_ref(dls));
        unsafe { ash_dev.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| Error::Msg(format!("vulkan allocate_descriptor_sets failed: {e:?}")))?
            .into_iter()
            .next()
            .map(|d| desc_set_handle = d);
        if desc_set_handle == vk::DescriptorSet::null() {
            return Err(Error::Msg("vulkan allocate_descriptor_sets empty".into()));
        }

        let buf_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|&b| vk::DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bi)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set_handle)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(bi))
            })
            .collect();
        unsafe { ash_dev.update_descriptor_sets(&writes, &[]) };

        let cmd_buf_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self._command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { ash_dev.allocate_command_buffers(&cmd_buf_info) }
            .map_err(|e| Error::Msg(format!("vulkan allocate_command_buffers failed: {e:?}")))?;
        let cmd = cmd_bufs[0];

        unsafe {
            ash_dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(|e| Error::Msg(format!("vulkan begin_command_buffer failed: {e:?}")))?;

        unsafe { ash_dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        unsafe {
            ash_dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self._pipeline_layout,
                0,
                &[desc_set_handle],
                &[],
            )
        };
        if !push.is_empty() {
            unsafe {
                ash_dev.cmd_push_constants(
                    cmd,
                    self._pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                )
            };
        }
        unsafe { ash_dev.cmd_dispatch(cmd, global.0, global.1, global.2) };
        unsafe { ash_dev.end_command_buffer(cmd) }
            .map_err(|e| Error::Msg(format!("vulkan end_command_buffer failed: {e:?}")))?;

        // Host access to a VkQueue is externally synchronized: hold the per-device
        // queue lock from submit through the matching wait so a concurrent dispatch
        // on a cloned device can't race the same queue.
        let _guard = unsafe { dev.with_queue_lock() };
        let submit_cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&submit_cmds);
        unsafe { ash_dev.queue_submit(dev.queue(), &[submit], vk::Fence::null()).map_err(|e| Error::Msg(format!("vulkan queue_submit failed: {e:?}")))? };
        unsafe { ash_dev.queue_wait_idle(dev.queue()) }
            .map_err(|e| Error::Msg(format!("vulkan queue_wait_idle failed: {e:?}")))?;
        Ok(())
    }
}

impl Drop for Dispatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self._command_pool, None);
            self.device.destroy_descriptor_pool(self._descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self._descriptor_set_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self._pipeline_layout, None);
            self.device.destroy_shader_module(self._shader_module, None);
        }
    }
}

const ELEM_WORKGROUP: u32 = 256;

/// Elementwise dispatch helper: launches `ceil(n/256)` 1D workgroups (never
/// more invocations than elements — the shaders bounds-check against `N`).
/// `push_size` is the byte size of `push`'s layout
/// (4 for `{uint N}`, 12 for `{uint N; float a; float b}`); it also drives the
/// pipeline layout's push-constant range.
fn dispatch_1buf(
    dev: &VulkanDevice,
    spirv: &[u32],
    inp: &VulkanStorage,
    out: &VulkanStorage,
    n: usize,
    push: &[u8],
    push_size: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let d = Dispatch::new(dev, spirv, 2, push_size)?;
    let groups = ((n as u32) + ELEM_WORKGROUP - 1) / ELEM_WORKGROUP;
    d.run(dev, &[inp.buffer, out.buffer], (ELEM_WORKGROUP, 1, 1), (groups, 1, 1), push)
}

fn dispatch_2buf(
    dev: &VulkanDevice,
    spirv: &[u32],
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    out: &VulkanStorage,
    n: usize,
    push: &[u8],
    push_size: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let d = Dispatch::new(dev, spirv, 3, push_size)?;
    let groups = ((n as u32) + ELEM_WORKGROUP - 1) / ELEM_WORKGROUP;
    d.run(
        dev,
        &[lhs.buffer, rhs.buffer, out.buffer],
        (ELEM_WORKGROUP, 1, 1),
        (groups, 1, 1),
        push,
    )
}

/// out[i] = a*inp[i] + b.  Push layout `{uint N; float a; float b}`.
pub fn run_affine(
    dev: &VulkanDevice,
    inp: &VulkanStorage,
    n: usize,
    a: f32,
    b: f32,
) -> Result<VulkanStorage> {
    let spirv = glsl_to_spirv(GLSL_AFFINE, "main")?;
    let out = unsafe { dev.alloc_buffer(n * 4, crate::DType::F32, n) }?;
    let mut push = Vec::with_capacity(12);
    push.extend_from_slice(&(n as u32).to_le_bytes());
    push.extend_from_slice(&a.to_le_bytes());
    push.extend_from_slice(&b.to_le_bytes());
    dispatch_1buf(dev, &spirv, inp, &out, n, &push, 12)?;
    Ok(out)
}

/// out[i] = op(inp[i]).  Push layout `{uint N}`.
pub fn run_unary(dev: &VulkanDevice, name: &str, inp: &VulkanStorage, n: usize) -> Result<VulkanStorage> {
    let expr = match name {
        "exp" => "exp(x)",
        "sin" => "sin(x)",
        "cos" => "cos(x)",
        "tan" => "tan(x)",
        "sqrt" => "sqrt(x)",
        "abs" => "abs(x)",
        "neg" => "-x",
        "ln" => "log(x)",
        "sqr" => "x * x",
        _ => return Err(Error::Msg(format!("vulkan unary {name} not wired"))),
    };
    let src = GLSL_UNARY.replace("{EXPR}", expr);
    let spirv = glsl_to_spirv(&src, "main")?;
    let out = unsafe { dev.alloc_buffer(n * 4, crate::DType::F32, n) }?;
    let push = (n as u32).to_le_bytes().to_vec();
    dispatch_1buf(dev, &spirv, inp, &out, n, &push, 4)?;
    Ok(out)
}

/// out[i] = op(lhs[i], rhs[i]).  Push layout `{uint N}`.
pub fn run_binary(
    dev: &VulkanDevice,
    name: &str,
    lhs: &VulkanStorage,
    rhs: &VulkanStorage,
    n: usize,
) -> Result<VulkanStorage> {
    let expr = match name {
        "add" => "a + b",
        "mul" => "a * b",
        "sub" => "a - b",
        "div" => "a / b",
        "pow" => "pow(a, b)",
        _ => return Err(Error::Msg(format!("vulkan binary {name} not wired"))),
    };
    let src = GLSL_BINARY.replace("{EXPR}", expr);
    let spirv = glsl_to_spirv(&src, "main")?;
    let out = unsafe { dev.alloc_buffer(n * 4, crate::DType::F32, n) }?;
    let push = (n as u32).to_le_bytes().to_vec();
    dispatch_2buf(dev, &spirv, lhs, rhs, &out, n, &push, 4)?;
    Ok(out)
}

/// o(row-major m×n) = a(m×k) @ b(k×n), batched.
/// `batch` matrices run with the batch index on the workgroup Z axis. `bsk/bsn`
/// are b's layout strides along k and n (transposed/broadcast rhs handled
/// in-kernel); `ba/bb/bo` are the per-batch strides of a/b/o.
pub fn run_matmul(
    dev: &VulkanDevice,
    a: &VulkanStorage,
    b: &VulkanStorage,
    (batch, m, n, k): (usize, usize, usize, usize),
    bsk: usize,
    bsn: usize,
    ba: usize,
    bb: usize,
    bo: usize,
) -> Result<VulkanStorage> {
    let spirv = glsl_to_spirv(GLSL_MATMUL, "main")?;
    let out = unsafe {
        dev.alloc_buffer(m * n * batch * 4, crate::DType::F32, m * n * batch)
    }?;
    if m == 0 || n == 0 || k == 0 || batch == 0 {
        return Ok(out); // no work; buffer is empty/valid
    }
    let d = Dispatch::new(dev, &spirv, 3, 32)?;
    let push = [
        m as u32,
        n as u32,
        k as u32,
        bsk as u32,
        bsn as u32,
        ba as u32,
        bb as u32,
        bo as u32,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect::<Vec<u8>>();
    // Each 16x16 workgroup computes a 64x16 output tile (4 rows x 1 col per
    // thread): grid X = ceil(N/16), grid Y = ceil(M/64), grid Z = batch.
    let tile_n = 16u32;
    let tile_m = 64u32;
    let gx = ((n as u32) + tile_n - 1) / tile_n;
    let gy = ((m as u32) + tile_m - 1) / tile_m;
    d.run(
        dev,
        &[a.buffer, b.buffer, out.buffer],
        (16, 16, 1),
        (gx, gy, batch as u32),
        &push,
    )?;
    Ok(out)
}
/// Reduce the last dimension of a contiguous f32 tensor `x` shaped `(rows, cols)`
/// into a `(rows,)` output. `op` selects sum / max / mean.
pub fn run_reduce_last_dim(
    dev: &VulkanDevice,
    x: &VulkanStorage,
    rows: usize,
    cols: usize,
    op: ReduceLastDimOp,
) -> Result<VulkanStorage> {
    if rows == 0 {
        return unsafe { dev.alloc_buffer(0, crate::DType::F32, 0) };
    }
    // An empty row has no well-defined max (and the CPU path errors); keep the
    // kernel's max-from-first-element contract by rejecting cols == 0.
    if cols == 0 {
        if op == ReduceLastDimOp::Max {
            return Err(Error::Msg(
                "vulkan reduce max over an empty row has no identity".into(),
            ));
        }
        // Sum over an empty row is 0; allocate a zeroed output.
        let out = unsafe { dev.alloc_buffer(rows * 4, crate::DType::F32, rows) }?;
        let zeros = vec![0u8; rows * 4];
        unsafe { out.set_bytes(&zeros) }?;
        return Ok(out);
    }
    let spirv = glsl_to_spirv(GLSL_REDUCE_LASTDIM, "main")?;
    let out = unsafe { dev.alloc_buffer(rows * 4, crate::DType::F32, rows) }?;
    let d = Dispatch::new(dev, &spirv, 2, 12)?;
    let opval = match op {
        ReduceLastDimOp::Sum => 0u32,
        ReduceLastDimOp::Max => 1u32,
        ReduceLastDimOp::Mean => 2u32,
    };
    let push = [rows as u32, cols as u32, opval]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    // The workgroup is 64 threads wide, so one X workgroup already covers up to
    // 64 rows: dispatch ceil(rows/64) X workgroups (the shader bounds-checks the
    // final partial group) instead of one per row.
    let groups_x = ((rows as u32) + 63) / 64;
    d.run(
        dev,
        &[x.buffer, out.buffer],
        (64, 1, 1),
        (groups_x, 1, 1),
        &push,
    )?;
    Ok(out)
}
