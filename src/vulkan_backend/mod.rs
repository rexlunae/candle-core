//! Vulkan backend for candle-core.
//!
//! Enabled with the `vulkan` feature. Uses `ash` (the well-known Vulkan
//! binding) with its `loaded` feature, so `libvulkan.so` is pulled in at
//! runtime via `libloading` rather than hard-linked — the same "dependency-
//! lean" approach the OpenCL backend uses against `-lOpenCL`, and it lets a
//! missing Vulkan loader degrade to a clear runtime error instead of a link
//! failure.
//!
//! Bring-up target is an AMD Renoir iGPU (unified memory). On such hardware a
//! device buffer can be backed by a `HOST_VISIBLE | HOST_COHERENT` memory type,
//! so the M1 host<->device round-trip is a plain memcpy through the `vkMapMemory`
//! pointer (no staging-buffer dance). Compute kernels (M2) are compiled at
//! runtime from embedded GLSL via `naga` (pure Rust GLSL→SPIR-V), so no external
//! `glslc` is required to build or run.
#![allow(clippy::missing_safety_doc)]

pub mod shaders;

use ash::vk;
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

/// The `DeviceLocation::Vulkan` variant carries a `gpu_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

/// One Vulkan logical device + the queue + memory-type metadata it needs to
/// allocate and copy buffers, wrapped in an `Arc` so every clone / storage
/// sharing it keeps the ash handles alive.
// Note: no `#[derive(Debug)]` here — ash's `Entry`/`Instance`/`Device` don't
// implement `Debug`, so the whole handle set is printed opaque.
struct VulkanContext {
    /// The loaded `libvulkan.so` handle. `#[allow(dead_code)]`: it's never read
    /// directly, but keeping it here guarantees the loader stays loaded for the
    /// lifetime of the device/instance below (dropping it could unload the lib
    /// out from under an outstanding `Instance`/`Device` call).
    #[allow(dead_code)]
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    /// Serialises host access to `queue`. Vulkan defines host access to a
    /// `VkQueue` as externally synchronized, so every submit + wait (and
    /// `synchronize`) must hold this lock to keep concurrent tensor ops from
    /// racing the same queue.
    queue_lock: std::sync::Mutex<()>,
    /// Physical-device memory properties, retained so `alloc_buffer` can pick a
    /// HOST_VISIBLE|HOST_COHERENT type that is *also* set in each resource's
    /// `memory_type_bits` (compatibility is per-resource, not global).
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    #[allow(dead_code)]
    min_uniform_alignment: vk::DeviceSize,
}

unsafe impl Send for VulkanContext {}
unsafe impl Sync for VulkanContext {}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// A Vulkan device: an immutable `gpu_id` + the shared device/queue handles.
// No `Debug` derive: ash handles (Entry/Instance/Device) are not `Debug`.
pub struct VulkanDevice {
    gpu_id: usize,
    inner: std::sync::Arc<VulkanContext>,
}

impl Clone for VulkanDevice {
    fn clone(&self) -> Self {
        VulkanDevice {
            gpu_id: self.gpu_id,
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice").field("gpu_id", &self.gpu_id).finish()
    }
}

/// Connect to the `gpu_id`-th physical device and create a logical device over
/// it, choosing HOST_VISIBLE|HOST_COHERENT memory for the round-trip fast path.
fn init_vulkan(gpu_id: usize) -> Result<VulkanDevice> {
    // # Safety: `Entry::load` is unsafe in ash 0.38 (dlopens/dlsyms); the loaded
    // module lives as long as the returned `Entry`, which we own below.
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            return Err(Error::Msg(format!(
                "vulkan: failed to load libvulkan.so ({e}); is a Vulkan loader + ICD installed?"
            )));
        }
    };

    let app_name = std::ffi::CString::new("candle-vulkan").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_2);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(e) => {
            return Err(Error::Msg(format!(
                "vulkan: create_instance failed: {e:?}"
            )));
        }
    };

    let phys_list = match unsafe { instance.enumerate_physical_devices() } {
        Ok(ps) if !ps.is_empty() => ps,
        Ok(_) => {
            let _ = unsafe { instance.destroy_instance(None) };
            return Err(Error::Msg("vulkan: no physical device found".into()));
        }
        Err(e) => {
            let _ = unsafe { instance.destroy_instance(None) };
            return Err(Error::Msg(format!(
                "vulkan: enumerate_physical_devices failed: {e:?}"
            )));
        }
    };

    let physical = match phys_list.get(gpu_id) {
        Some(&p) => p,
        None => {
            let _ = unsafe { instance.destroy_instance(None) };
            return Err(Error::Msg(format!(
                "vulkan: gpu_id {gpu_id} out of range ({} physical device(s) available)",
                phys_list.len()
            )));
        }
    };
    let idx = gpu_id;

    let props = unsafe { instance.get_physical_device_properties(physical) };
    let mut device_name = [0u8; vk::MAX_PHYSICAL_DEVICE_NAME_SIZE];
    for (i, b) in props.device_name.iter().enumerate() {
        device_name[i] = *b as u8;
    }
    let name = String::from_utf8_lossy(&device_name)
        .trim_end_matches('\0')
        .to_string();

    // Find a compute-capable queue family.
    let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
    let mut queue_family_index = None;
    for (i, q) in queues.iter().enumerate() {
        if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            queue_family_index = Some(i as u32);
            break;
        }
    }
    let queue_family = queue_family_index.ok_or_else(|| {
        let _ = unsafe { instance.destroy_instance(None) };
        Error::Msg(format!("vulkan: device {name} has no compute queue"))
    })?;

    let queue_priorities = [1.0f32];
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities)];
    let device_create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);
    let device = match unsafe { instance.create_device(physical, &device_create_info, None) } {
        Ok(d) => d,
        Err(e) => {
            let _ = unsafe { instance.destroy_instance(None) };
            return Err(Error::Msg(format!(
                "vulkan: create_device on {name} failed: {e:?}"
            )));
        }
    };

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    // Retain the memory properties so alloc_buffer can pick a compatible
    // HOST_VISIBLE|HOST_COHERENT type per resource. Early-validate that at
    // least one such type exists (the unified-memory fast path we rely on).
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let has_host_coh = mem_props.memory_types.iter().any(|mt| {
        mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    });
    if !has_host_coh {
        let _ = unsafe {
            device.destroy_device(None);
            instance.destroy_instance(None)
        };
        return Err(Error::Msg(format!(
            "vulkan: {name} exposes no HOST_VISIBLE|HOST_COHERENT memory type \
             (needed for the unified-memory round-trip fast path)"
        )));
    }

    eprintln!("vulkan: connected to {name} (gpu {idx}), queue family {queue_family}");

    Ok(VulkanDevice {
        gpu_id,
        inner: std::sync::Arc::new(VulkanContext {
            entry,
            instance,
            device,
            queue,
            queue_family,
            queue_lock: std::sync::Mutex::new(()),
            memory_properties: mem_props,
            min_uniform_alignment: 0,
        }),
    })
}

impl VulkanDevice {
    pub fn new(gpu_id: usize) -> Result<Self> {
        init_vulkan(gpu_id)
    }

    pub fn new_with_stream(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    pub fn id(&self) -> DeviceId {
        DeviceId(self.gpu_id)
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    pub(crate) fn queue(&self) -> vk::Queue {
        self.inner.queue
    }

    /// Acquire the per-device queue lock for the duration of a submit+wait,
    /// returning a `MutexGuard` whose drop releases it.
    ///
    /// # Safety
    /// The caller must hold the returned guard for the whole span of Vulkan
    /// commands that touch `self.queue()` (e.g. `queue_submit` up to the
    /// matching `queue_wait_idle`), because host access to a `VkQueue` is
    /// externally synchronized.
    pub(crate) unsafe fn with_queue_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        self.inner.queue_lock.lock().expect("vulkan queue lock poisoned")
    }

    pub(crate) fn queue_family(&self) -> u32 {
        self.inner.queue_family
    }

    /// The HOST_VISIBLE|HOST_COHERENT memory-type index (exposed for the shader
    /// layer to reason about the round-trip fast path).
    #[allow(dead_code)]
    pub(crate) fn host_chain_mem_index(&self) -> u32 {
        self.inner
            .memory_properties
            .memory_types
            .iter()
            .position(|mt| {
                mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .unwrap_or(0) as u32
    }

    /// Allocate a `VulkanStorage` device buffer of `bytes`
    /// (HOST_VISIBLE|HOST_COHERENT) and map it, returning a pointer for direct
    /// CPU reads/writes.
    ///
    /// # Safety
    /// Caller must ensure no concurrent use of the returned handle across
    /// threads that could race on the same buffer.
    pub(crate) unsafe fn alloc_buffer(
        &self,
        bytes: usize,
        dtype: DType,
        numel: usize,
    ) -> Result<VulkanStorage> {
        // Vulkan forbids zero-sized buffers; represent a 0-element tensor with a
        // minimal backing allocation while keeping the logical capacity/numel.
        let backing = if bytes == 0 { 1 } else { bytes };
        let buffer = unsafe {
            self.inner
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(backing as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(|e| Error::Msg(format!("vulkan create_buffer failed: {e:?}")))
        }?;

        let mem_req = unsafe { self.inner.device.get_buffer_memory_requirements(buffer) };
        // Choose a HOST_VISIBLE|HOST_COHERENT type that is ALSO compatible with
        // this specific buffer (its `memory_type_bits`).
        let mem_type = self
            .inner
            .memory_properties
            .memory_types
            .iter()
            .enumerate()
            .find(|(i, mt)| {
                (mem_req.memory_type_bits & (1 << i)) != 0
                    && mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
                    && mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .map(|(i, _)| i as u32)
            .ok_or_else(|| {
                let _ = unsafe { self.inner.device.destroy_buffer(buffer, None) };
                Error::Msg(format!(
                    "vulkan: no compatible HOST_VISIBLE|HOST_COHERENT memory type for size {bytes} (mask {:b}, types {})",
                    mem_req.memory_type_bits,
                    self.inner.memory_properties.memory_types.len()
                ))
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { self.inner.device.allocate_memory(&alloc_info, None) }.map_err(
            |e| {
                let _ = unsafe { self.inner.device.destroy_buffer(buffer, None) };
                Error::Msg(format!("vulkan allocate_memory failed: {e:?}"))
            },
        )?;
        unsafe { self.inner.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| {
            let _ = unsafe { self.inner.device.destroy_buffer(buffer, None) };
            let _ = unsafe { self.inner.device.free_memory(memory, None) };
            Error::Msg(format!("vulkan bind_buffer_memory failed: {e:?}"))
        })?;
        let mapped = unsafe {
            self.inner
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    let _ = self.inner.device.destroy_buffer(buffer, None);
                    let _ = self.inner.device.free_memory(memory, None);
                    Error::Msg(format!("vulkan map_memory failed: {e:?}"))
                })
        }?;

        Ok(VulkanStorage {
            buffer,
            memory,
            mapped: mapped as *mut u8,
            capacity_bytes: bytes, // logical capacity (0 for empty tensors)
            dtype,
            numel,
            device: self.clone(),
        })
    }
}

/// A Vulkan storage: a raw `VkBuffer` (+ its mapped host pointer) + dtype +
/// element count. Backed by host-visible+coherent memory for the unified-memory
/// round-trip fast path.
// No `Debug` derive: `VulkanDevice` holds ash handles that are not `Debug`.
pub struct VulkanStorage {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub mapped: *mut u8,
    pub capacity_bytes: usize,
    pub dtype: DType,
    pub numel: usize,
    pub device: VulkanDevice,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanStorage")
            .field("capacity_bytes", &self.capacity_bytes)
            .field("dtype", &self.dtype)
            .field("numel", &self.numel)
            .field("device", &self.device)
            .finish()
    }
}

unsafe impl Send for VulkanStorage {}
unsafe impl Sync for VulkanStorage {}

impl Drop for VulkanStorage {
    fn drop(&mut self) {
        let dev = &self.device.inner.device;
        unsafe {
            dev.unmap_memory(self.memory);
            dev.destroy_buffer(self.buffer, None);
            dev.free_memory(self.memory, None);
        }
    }
}

impl VulkanStorage {
    pub fn transfer_to_device(&self, dst: &VulkanDevice) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        dst.storage_from_cpu_storage(&cpu)
    }

    pub fn from_vec<T: crate::WithDType>(slice: Vec<T>, device: &VulkanDevice) -> Result<Self> {
        let dtype = T::DTYPE;
        let bytes = dtype
            .size_in_bytes()
            .checked_mul(slice.len())
            .ok_or_else(|| Error::Msg("vulkan: overflow in storage size".into()))?;
        // # Safety: `slice.as_ptr()` points to `bytes` valid bytes.
        let data = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) };
        let storage = unsafe { device.alloc_buffer(bytes, dtype, slice.len()) }?;
        unsafe { storage.set_bytes(data) }?;
        Ok(storage)
    }

    /// Copy `src` bytes into the mapped host-visible buffer.
    ///
    /// # Safety
    /// `src` must be at most `self.capacity_bytes` long.
    pub(crate) unsafe fn set_bytes(&self, src: &[u8]) -> Result<()> {
        if src.len() > self.capacity_bytes {
            return Err(Error::Msg(format!(
                "vulkan set_bytes: {} bytes exceed capacity {}",
                src.len(),
                self.capacity_bytes
            )));
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(self.mapped as *mut u8, src.len()) };
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Read the mapped host-visible buffer bytes into a `Vec<u8>`.
    ///
    /// # Safety
    /// `bytes` must be <= `self.capacity_bytes`.
    pub(crate) unsafe fn get_bytes(&self, bytes: usize) -> Vec<u8> {
        let n = bytes.min(self.capacity_bytes);
        let mut out = vec![0u8; n];
        let src = unsafe { std::slice::from_raw_parts(self.mapped as *const u8, n) };
        out.copy_from_slice(src);
        out
    }
}

// --- Native-kernel gating helpers (mirror the OpenCL backend). ---
const VULKAN_NATIVE_MAX_DIM: usize = i32::MAX as usize;

fn native_ok(layout: &Layout, numel: usize) -> bool {
    layout.is_contiguous()
        && layout.start_offset() == 0
        && layout.shape().elem_count() == numel
        && numel <= VULKAN_NATIVE_MAX_DIM
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        self.device.storage_from_cpu_storage(&cpu)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(s, layout)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&cpu)?;
        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let bytes = self.dtype.size_in_bytes() * self.numel;
        let raw = unsafe { self.get_bytes(bytes) };
        Ok(match self.dtype {
            DType::U8 => {
                let mut v = vec![0u8; self.numel];
                v.copy_from_slice(&raw);
                CpuStorage::U8(v)
            }
            DType::U32 => CpuStorage::U32(transmute_bytes(&raw, self.numel)),
            DType::I16 => CpuStorage::I16(transmute_bytes(&raw, self.numel)),
            DType::I32 => CpuStorage::I32(transmute_bytes(&raw, self.numel)),
            DType::I64 => CpuStorage::I64(transmute_bytes(&raw, self.numel)),
            DType::F32 => CpuStorage::F32(transmute_bytes(&raw, self.numel)),
            DType::F64 => CpuStorage::F64(transmute_bytes(&raw, self.numel)),
            _ => {
                return Err(Error::Msg(format!(
                    "vulkan to_cpu_storage: dtype {:?} not supported in M1",
                    self.dtype
                )))
            }
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if shaders::native_enabled() && native_ok(layout, self.numel) && self.dtype == DType::F32 {
            let n = self.numel;
            match shaders::run_affine(&self.device, self, n, mul as f32, add as f32) {
                Ok(out) => {
                    shaders::note_native_exec();
                    return Ok(out);
                }
                Err(e) => shaders::log_native_fallback("affine", &e),
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.affine(layout, mul, add)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.powf(layout, e)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.elu(layout, alpha)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        // Native fast path: reduce the *last* dim of a contiguous F32 tensor to a
        // 1-D result. This is the primitive softmax (max, then sum) and RMSNorm
        // (sum of squares / mean) need. SUM/MAX are supported directly; MIN is
        // handled by negating around a MAX. Mean is folded into the reshape-the
        // -result caller on CPU, so we only do raw reductions here.
        if shaders::native_enabled()
            && self.dtype == DType::F32
            && layout.is_contiguous()
            && layout.start_offset() == 0
            && s.len() == 1
            && s[0] == layout.shape().dims().len() - 1 // last dim
            && layout.shape().rank() >= 1
        {
            let dims = layout.shape().dims();
            let cols = dims[dims.len() - 1];
            let rows = if dims.len() == 1 { 1 } else { dims[..dims.len() - 1].iter().product() };
            let reduced_op = match op {
                ReduceOp::Sum => Some(shaders::ReduceLastDimOp::Sum),
                ReduceOp::Max => Some(shaders::ReduceLastDimOp::Max),
                // Min == -max(-x); emit a max here and let the caller negate? Not
                // composable, so fall back to CPU for Min.
                _ => None,
            };
            if let Some(rop) = reduced_op {
                match shaders::run_reduce_last_dim(&self.device, self, rows, cols, rop) {
                    Ok(storage) => {
                        shaders::note_native_exec();
                        return Ok(storage);
                    }
                    Err(e) => shaders::log_native_fallback("reduce_op", &e),
                }
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.reduce_op(op, layout, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.cmp(op, &rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.to_dtype(layout, dtype)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        if shaders::native_enabled()
            && native_ok(layout, self.numel)
            && self.dtype == DType::F32
            && shaders::has_unary(B::NAME)
        {
            let n = self.numel;
            match shaders::run_unary(&self.device, B::NAME, self, n) {
                Ok(out) => {
                    shaders::note_native_exec();
                    return Ok(out);
                }
                Err(e) => shaders::log_native_fallback(B::NAME, &e),
            }
        }
        let cpu = self.to_cpu_storage()?;
        let out = cpu.unary_impl::<B>(layout)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        if shaders::native_enabled()
            && native_ok(lhs_l, self.numel)
            && native_ok(rhs_l, rhs.numel)
            && self.dtype == DType::F32
            && rhs.dtype == DType::F32
            && self.numel == rhs.numel
            && shaders::has_binary(B::NAME)
        {
            let n = self.numel;
            match shaders::run_binary(&self.device, B::NAME, self, rhs, n) {
                Ok(out) => {
                    shaders::note_native_exec();
                    return Ok(out);
                }
                Err(e) => shaders::log_native_fallback(B::NAME, &e),
            }
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.binary_impl::<B>(&rhs, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let cond = self.to_cpu_storage()?;
        let t = t.to_cpu_storage()?;
        let f = f.to_cpu_storage()?;
        let out = cond.where_cond(layout, &t, t_l, &f, f_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose1d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let inp = self.to_cpu_storage()?;
        let kernel = kernel.to_cpu_storage()?;
        let out = inp.conv_transpose2d(l, &kernel, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.index_select(&ids, l, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let out = src.gather(l, &ids, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        tgt.scatter_add_set(l, &ids, ids_l, &src, src_l, dim)?;
        let dev = self.device.clone();
        *self = dev.storage_from_cpu_storage(&tgt)?;
        Ok(())
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let tgt = self.to_cpu_storage()?;
        let ids = ids.to_cpu_storage()?;
        let src = src.to_cpu_storage()?;
        let out = tgt.index_add(l, &ids, ids_l, &src, src_l, dim)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if shaders::native_enabled()
            && native_ok(lhs_l, self.numel)
            && rhs_l.start_offset() == 0
            && self.dtype == DType::F32
            && rhs.dtype == DType::F32
        {
            let (batch, m, n, k) = bmnk;
            if batch <= VULKAN_NATIVE_MAX_DIM
                && m <= VULKAN_NATIVE_MAX_DIM
                && n <= VULKAN_NATIVE_MAX_DIM
                && k <= VULKAN_NATIVE_MAX_DIM
                && rhs.numel <= VULKAN_NATIVE_MAX_DIM
            {
                // The kernel reads B as b[bb*bz + k*bsk + col*bsn]; bsk/bsn are
                // the RHS layout strides along k/n, so a transposed weight
                // ([1,K]) or the attention KV broadcast ([0,1]) runs on-device.
                // ba/bb/bo are the per-batch strides of A/B/o (0 => broadcast).
                // LHS stays row-major (confirmed contiguous in the profile).
                let rstride = rhs_l.stride();
                let rd = rstride.len();
                if rd >= 2 {
                    let bsn = rstride[rd - 1];
                    let bsk = rstride[rd - 2];
                    // `batch` flattens every leading axis, so for rank>=4 the
                    // per-matrix stride is the *matrix element count*, not the
                    // first axis's stride. LHS is contiguous (native_ok), so its
                    // matrices are laid out linearly at m*k.
                    let ba = m * k;
                    // RHS: accept native only when its batch is linear — the
                    // innermost batch-axis stride equals the k*n matrix size
                    // (contiguous / transposed storage both satisfy this). A
                    // broadcast batch gives 0. Any other layout falls back to CPU
                    // rather than indexing the wrong matrix.
                    let bb = if rd >= 3 {
                        let inner = rstride[rd - 3];
                        if inner == k * n { inner } else { 0 }
                    } else {
                        0 // 2D RHS broadcast across all batch groups
                    };
                    let bo = m * n;
                    match shaders::run_matmul(
                        &self.device, self, rhs, (batch, m, n, k), bsk, bsn, ba, bb, bo,
                    ) {
                        Ok(out) => {
                            shaders::note_native_exec();
                            return Ok(out);
                        }
                        Err(e) => shaders::log_native_fallback("matmul", &e),
                    }
                }
            }
        }
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        let out = lhs.matmul(&rhs, bmnk, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let src = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;
        let dev = dst.device.clone();
        *dst = dev.storage_from_cpu_storage(&dst_cpu)?;
        Ok(())
    }

    fn avg_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.avg_pool2d(layout, k, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn max_pool2d(&self, layout: &Layout, k: (usize, usize), s: (usize, usize)) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.max_pool2d(layout, k, s)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_nearest1d(layout, sz)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_nearest2d(layout, h, w)?;
        self.device.storage_from_cpu_storage(&out)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let out = cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)?;
        self.device.storage_from_cpu_storage(&out)
    }
}

/// Reinterpret `raw` (a byte buffer of `numel * size` bytes) as a `Vec<T>` by
/// copying. The caller must guarantee `raw.len() == numel * size_of::<T>()`
/// and that the bytes are a valid bit pattern for `T`.
fn transmute_bytes<T: Copy>(raw: &[u8], numel: usize) -> Vec<T> {
    let t = std::mem::size_of::<T>();
    debug_assert_eq!(raw.len(), numel * t);
    let mut v = vec![unsafe { std::mem::zeroed() }; numel];
    let dst = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, numel * t) };
    dst.copy_from_slice(raw);
    v
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        Self::new(gpu_id)
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("vulkan set_seed not implemented yet (M2)")
    }

    fn get_current_seed(&self) -> Result<u64> {
        crate::bail!("vulkan get_current_seed not implemented yet (M2)")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan { gpu_id: self.gpu_id }
    }

    fn same_device(&self, other: &Self) -> bool {
        // Two devices are only "the same" when they share the exact same logical
        // device handles (i.e. are clones of one another). Comparing only gpu_id
        // would let two independent contexts look equal and route cross-context
        // operations down the native path, silently failing their kernels.
        self.gpu_id == other.gpu_id && std::sync::Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        let storage = unsafe { self.alloc_uninit(shape, dtype) }?;
        if numel > 0 && dtype.size_in_bytes() > 0 {
            let bytes = numel * dtype.size_in_bytes();
            let zeros = vec![0u8; bytes];
            unsafe { storage.set_bytes(&zeros) }?;
        }
        Ok(storage)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let numel = shape.elem_count();
        if dtype.size_in_bytes() == 0 {
            return Err(Error::Msg(
                "vulkan alloc_uninit: unsupported (sub-byte) dtype".into(),
            ));
        }
        let bytes = numel * dtype.size_in_bytes();
        unsafe { self.alloc_buffer(bytes, dtype, numel) }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        VulkanStorage::from_vec(s.to_vec(), self)
    }

    fn storage_from_cpu_storage(&self, cpu: &CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage_owned(cpu.clone())
    }

    fn storage_from_cpu_storage_owned(&self, cpu: CpuStorage) -> Result<Self::Storage> {
        let dtype = cpu.dtype();
        match &cpu {
            CpuStorage::U8(v) => VulkanStorage::from_vec(v.clone(), self),
            CpuStorage::U32(v) => VulkanStorage::from_vec(v.clone(), self),
            CpuStorage::I16(v) => VulkanStorage::from_vec(v.clone(), self),
            CpuStorage::I32(v) => VulkanStorage::from_vec(v.clone(), self),
            CpuStorage::I64(v) => VulkanStorage::from_vec(v.clone(), self),
            CpuStorage::F32(v) => VulkanStorage::from_vec(v.clone(), self),
            CpuStorage::F64(v) => VulkanStorage::from_vec(v.clone(), self),
            _ => Err(Error::Msg(format!(
                "vulkan storage_from_cpu_storage: dtype {:?} not supported in M1",
                dtype
            ))),
        }
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        crate::bail!("vulkan rand_uniform not implemented yet (M2)")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        crate::bail!("vulkan rand_normal not implemented yet (M2)")
    }

    fn synchronize(&self) -> Result<()> {
        // Hold the queue lock so we don't race a concurrent dispatch's submit.
        let _guard = unsafe { self.with_queue_lock() };
        unsafe {
            self.inner
                .device
                .queue_wait_idle(self.inner.queue)
                .map_err(|e| Error::Msg(format!("vulkan queue_wait_idle failed: {e:?}")))
        }
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    /// M1 round-trip: write f32 to a host-visible Vulkan buffer and read it back.
    /// Run on the AMD Renoir iGPU (or any Vulkan host) with:
    ///   cargo test --features vulkan -p candle-core -- --ignored vulkan
    #[test]
    #[ignore]
    fn f32_roundtrip_via_tensor() -> crate::Result<()> {
        let dev = match crate::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping vulkan round-trip: {e}");
                return Ok(());
            }
        };
        let dev = Device::Vulkan(dev);
        let a = Tensor::from_vec(vec![1.5f32, 2.0, 3.0, -4.5], (4,), &dev)?;
        let cpu = a.to_device(&Device::Cpu)?;
        let v = cpu.to_vec1::<f32>()?;
        assert_eq!(v, vec![1.5f32, 2.0, 3.0, -4.5]);
        Ok(())
    }

    /// M2 parity: affine/unary/binary/matmul computed on Cpu and Vulkan native
    /// kernels (enabled only when JOSHUA_VULKAN_NATIVE=1) must agree.
    #[test]
    #[ignore]
    fn vulkan_parity_native() -> crate::Result<()> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);

        let dev = match crate::VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping vulkan parity: {e}");
                return Ok(());
            }
        };
        let dev = Device::Vulkan(dev);

        let native = shaders::native_enabled();
        eprintln!("JOSHUA_VULKAN_NATIVE={native}");
        let exec0 = shaders::native_exec_count();

        macro_rules! gen {
            ($n:expr) => {{
                (0..$n)
                    .map(|_| rng.gen_range(-2.0f32..2.0))
                    .collect::<Vec<f32>>()
            }};
        }

        fn approx(a: &[f32], b: &[f32], tol: f32, what: &str) {
            assert_eq!(a.len(), b.len(), "{what}: length");
            let mut worst = 0.0f32;
            for i in 0..a.len() {
                let mut d = (a[i] - b[i]).abs();
                if a[i].abs() + b[i].abs() > 1.0 {
                    d /= (a[i].abs() + b[i].abs()).max(1e-6);
                }
                if !d.is_finite() || d > worst {
                    worst = d;
                }
            }
            eprintln!("  {what}: worst rel diff = {worst:.3e} (len {})", a.len());
            assert!(worst < tol, "{what}: parity FAILED worst={worst:.3e}");
        }

        let n = 4096;
        let cpu_v = gen!(n);
        let mul = 1.5f32;
        let add = -0.25f32;
        let cpu = Tensor::from_vec(cpu_v.clone(), (n,), &Device::Cpu)?;
        let got_cpu = cpu.affine(f64::from(mul), f64::from(add))?;
        let vk = Tensor::from_vec(cpu_v, (n,), &dev)?;
        let got_vk = vk
            .affine(f64::from(mul), f64::from(add))?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        approx(&got_cpu.to_vec1::<f32>()?, &got_vk, 1e-3, "affine");

        let cpu_v = gen!(n);
        let cpu = Tensor::from_vec(cpu_v.clone(), (n,), &Device::Cpu)?;
        let got_cpu = cpu.exp()?;
        let vk = Tensor::from_vec(cpu_v, (n,), &dev)?;
        let got_vk = vk.exp()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        approx(&got_cpu.to_vec1::<f32>()?, &got_vk, 1e-3, "exp");

        let a_v = gen!(n);
        let b_v = gen!(n);
        let ac = Tensor::from_vec(a_v.clone(), (n,), &Device::Cpu)?;
        let bc = Tensor::from_vec(b_v.clone(), (n,), &Device::Cpu)?;
        let got_cpu = ac.add(&bc)?;
        let ao = Tensor::from_vec(a_v, (n,), &dev)?;
        let bo = Tensor::from_vec(b_v, (n,), &dev)?;
        let got_vk = ao.add(&bo)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        approx(&got_cpu.to_vec1::<f32>()?, &got_vk, 1e-3, "add");

        let (mm, _mk, mn, mk2) = (37usize, 64usize, 51usize, 64usize);
        let a_v = (0..mm * mk2).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect::<Vec<f32>>();
        let b_v = (0..mk2 * mn).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect::<Vec<f32>>();
        let ac = Tensor::from_vec(a_v.clone(), (mm, mk2), &Device::Cpu)?;
        let bc = Tensor::from_vec(b_v.clone(), (mk2, mn), &Device::Cpu)?;
        let got_cpu = ac.matmul(&bc)?;
        let ao = Tensor::from_vec(a_v, (mm, mk2), &dev)?;
        let bo = Tensor::from_vec(b_v, (mk2, mn), &dev)?;
        let got_vk = ao
            .matmul(&bo)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        approx(
            &got_cpu.flatten_all()?.to_vec1::<f32>()?,
            &got_vk,
            1e-2,
            "matmul(37,64,51)",
        );

        // Transposed-RHS matmul (weight-transpose pattern): ao @ (w_t.t()) where
        // w_t is (n,k) contiguous -> a (k,n) non-contiguous transposed view.
        // Exercises the kernel's bsk/bsn stride path.
        let (m2, k2, n2) = (23usize, 48usize, 36usize);
        let a2_v = (0..m2 * k2).map(|i| ((i % 5) as f32 - 2.0) * 0.4).collect::<Vec<f32>>();
        let w_v = (0..n2 * k2).map(|i| ((i % 9) as f32 - 4.0) * 0.3).collect::<Vec<f32>>(); // (n,k) row-major
        let ac2 = Tensor::from_vec(a2_v.clone(), (m2, k2), &Device::Cpu)?;
        let wc = Tensor::from_vec(w_v.clone(), (n2, k2), &Device::Cpu)?.t()?; // (k,n) transposed view
        let got_cpu2 = ac2.matmul(&wc)?;
        let a2o = Tensor::from_vec(a2_v, (m2, k2), &dev)?;
        let wo = Tensor::from_vec(w_v, (n2, k2), &dev)?.t()?;
        let got_vk2 = a2o
            .matmul(&wo)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        approx(
            &got_cpu2.flatten_all()?.to_vec1::<f32>()?,
            &got_vk2,
            1e-2,
            "matmul_transposed_rhs(23,48,36)",
        );

        // Batched matmul (batch>1): a (8,37,64) @ b (8,64,51).
        let (bsz, m3, k3, n3) = (8usize, 37usize, 64usize, 51usize);
        let a3_v = (0..bsz * m3 * k3).map(|i| ((i % 6) as f32 - 3.0) * 0.5).collect::<Vec<f32>>();
        let b3_v = (0..bsz * k3 * n3).map(|i| ((i % 10) as f32 - 5.0) * 0.25).collect::<Vec<f32>>();
        let a3c = Tensor::from_vec(a3_v.clone(), (bsz, m3, k3), &Device::Cpu)?;
        let b3c = Tensor::from_vec(b3_v.clone(), (bsz, k3, n3), &Device::Cpu)?;
        let got_cpu3 = a3c.matmul(&b3c)?;
        let a3o = Tensor::from_vec(a3_v, (bsz, m3, k3), &dev)?;
        let b3o = Tensor::from_vec(b3_v, (bsz, k3, n3), &dev)?;
        let got_vk3 = a3o
            .matmul(&b3o)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        approx(
            &got_cpu3.flatten_all()?.to_vec1::<f32>()?,
            &got_vk3,
            1e-2,
            "matmul_batched(8,37,64@64x51)",
        );

        // Rank-4 batched matmul (Devin): lhs [1,8,m,k] @ rhs [1,8,k,n].
        let a4_v = (0..1 * 8 * 64 * 512).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect::<Vec<f32>>();
        let b4_v = (0..1 * 8 * 512 * 51).map(|i| ((i % 11) as f32 - 5.0) * 0.25).collect::<Vec<f32>>();
        let a4c = Tensor::from_vec(a4_v.clone(), (1, 8, 64, 512), &Device::Cpu)?;
        let b4c = Tensor::from_vec(b4_v.clone(), (1, 8, 512, 51), &Device::Cpu)?;
        let got_cpu4 = a4c.matmul(&b4c)?;
        let a4o = Tensor::from_vec(a4_v, (1, 8, 64, 512), &dev)?;
        let b4o = Tensor::from_vec(b4_v, (1, 8, 512, 51), &dev)?;
        let got_vk4 = a4o
            .matmul(&b4o)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        approx(
            &got_cpu4.flatten_all()?.to_vec1::<f32>()?,
            &got_vk4,
            1e-2,
            "matmul_rank4(1,8,64,512)",
        );

        // Rank-4 with two non-unit batch axes: [2,3,m,k] @ [2,3,k,n].
        let a5_v = (0..2 * 3 * 16 * 24).map(|i| ((i % 9) as f32 - 4.0) * 0.3).collect::<Vec<f32>>();
        let b5_v = (0..2 * 3 * 24 * 20).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect::<Vec<f32>>();
        let a5c = Tensor::from_vec(a5_v.clone(), (2, 3, 16, 24), &Device::Cpu)?;
        let b5c = Tensor::from_vec(b5_v.clone(), (2, 3, 24, 20), &Device::Cpu)?;
        let got_cpu5 = a5c.matmul(&b5c)?;
        let a5o = Tensor::from_vec(a5_v, (2, 3, 16, 24), &dev)?;
        let b5o = Tensor::from_vec(b5_v, (2, 3, 24, 20), &dev)?;
        let got_vk5 = a5o
            .matmul(&b5o)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        approx(
            &got_cpu5.flatten_all()?.to_vec1::<f32>()?,
            &got_vk5,
            1e-2,
            "matmul_rank4_multi_axis(2,3,16,24)",
        );

        // Last-dim reduction (softmax/RMSNorm primitive): sum and max over rows.
        let (rr, cc) = (128usize, 64usize);
        let r_v = (0..rr * cc).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect::<Vec<f32>>();
        let redux_base = shaders::native_exec_count();
        let rc = Tensor::from_vec(r_v.clone(), (rr, cc), &Device::Cpu)?;
        let got_sum_cpu = rc.sum(1)?;
        let got_max_cpu = rc.max(1)?;
        let ro = Tensor::from_vec(r_v, (rr, cc), &dev)?;
        let got_sum_vk = ro.sum(1)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let got_max_vk = ro.max(1)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        approx(
            &got_sum_cpu.flatten_all()?.to_vec1::<f32>()?,
            &got_sum_vk,
            1e-2,
            "reduce_sum_lastdim(128,64)",
        );
        approx(
            &got_max_cpu.flatten_all()?.to_vec1::<f32>()?,
            &got_max_vk,
            1e-2,
            "reduce_max_lastdim(128,64)",
        );

        // The reductions must run natively (not silently fall back to CPU).
        if native {
            let redux = shaders::native_exec_count() - redux_base;
            assert!(
                redux >= 2,
                "JOSHUA_VULKAN_NATIVE was set but only {redux} of the 2 reductions ran natively"
            );
        }

        if native {
            let exec = shaders::native_exec_count() - exec0;
            eprintln!("vulkan parity: native kernel executions this test = {exec}");
            assert!(
                exec >= 4,
                "JOSHUA_VULKAN_NATIVE was set but only {exec} native kernels ran"
            );
        }

        eprintln!("vulkan parity: all wired ops OK");
        Ok(())
    }
}
