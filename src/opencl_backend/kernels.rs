//! Native OpenCL compute kernels (M5).
//!
//! M1-M4 round-trip dense ops through the CPU (correct, no device compute).
//! This adds real on-device kernels: one OpenCL program is compiled once per
//! device (cached in a global map keyed by context) and hot dense ops
//! (affine / elementwise / f32 GEMM) run as ND-range kernels on the iGPU.
//! Opt-in via JOSHUA_OPENCL_NATIVE; only used for contiguous f32 buffers.

use crate::{Error, Result};

extern "C" {
    fn clCreateProgramWithSource(context: usize, count: u32, strings: *const *const std::ffi::c_void, lengths: *const usize, err: *mut i32) -> usize;
    fn clBuildProgram(program: usize, ndev: u32, devices: *const usize, opts: *const std::ffi::c_void, cb: *const std::ffi::c_void, ud: *const std::ffi::c_void) -> i32;
    fn clCreateKernel(program: usize, name: *const std::ffi::c_void, err: *mut i32) -> usize;
    fn clSetKernelArg(kernel: usize, index: u32, size: usize, value: *const std::ffi::c_void) -> i32;
    fn clEnqueueNDRangeKernel(q: usize, kernel: usize, dim: u32, off: *const usize, gsz: *const usize, lsz: *const usize, nev: u32, ev: *const usize, e: *mut usize) -> i32;
    fn clReleaseProgram(program: usize) -> i32;
    fn clReleaseKernel(kernel: usize) -> i32;
}
const CL_SUCCESS: i32 = 0;

const KERNEL_SRC: &str = "
__kernel void kaffine(__global const float* x, __global float* o, int n, float mul, float add) {
    int i = get_global_id(0); if (i < n) o[i] = x[i] * mul + add;
}
__kernel void kexp(__global const float* x, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = exp(x[i]); }
__kernel void klog(__global const float* x, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = log(x[i]); }
__kernel void ksqrt(__global const float* x, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = sqrt(x[i]); }
__kernel void ksqr(__global const float* x, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = x[i] * x[i]; }
__kernel void kneg(__global const float* x, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = -x[i]; }
__kernel void krecip(__global const float* x, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = 1.0f / x[i]; }
__kernel void kadd(__global const float* a, __global const float* b, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = a[i] + b[i]; }
__kernel void ksub(__global const float* a, __global const float* b, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = a[i] - b[i]; }
__kernel void kmul(__global const float* a, __global const float* b, __global float* o, int n) { int i = get_global_id(0); if (i < n) o[i] = a[i] * b[i]; }
// Flat GEMM: one work-item per output element (fast on the Renoir iGPU:
// ~2ms for a decode matmul, and small banded dispatches stay well under the
// amdgpu ring-timeout watchdog). `row0` lets the caller split a large (M x N)
// GEMM into bands so no single submission lingers long enough to be killed.
__kernel void kmatmul(__global const float* a, __global const float* b, __global float* o, int M, int N, int K, int row0) {
    int row = row0 + get_global_id(0), col = get_global_id(1);
    if (row < row0 + M && col < N) { float acc = 0.0f; for (int k = 0; k < K; k++) acc += a[row*K+k] * b[k*N+col]; o[row*N+col] = acc; }
}
";

fn err(code: i32, op: &str) -> Error {
    Error::Msg(format!("opencl kernel {op} failed with status {code}"))
}

/// Compiled OpenCL program handle for one context.
pub struct KernelProg {
    pub program: usize,
}
impl Drop for KernelProg {
    fn drop(&mut self) {
        if self.program != 0 {
            unsafe { clReleaseProgram(self.program) };
        }
    }
}

/// Global per-context cache of compiled programs (compile once per process),
/// using the same OnceLock<Mutex<T>> idiom as the CUDA backend.
static PROGS: std::sync::OnceLock<std::sync::Mutex<Vec<(usize, KernelProg)>>> = std::sync::OnceLock::new();

/// Compile (or fetch cached) the kernel program for `(context, device_id)`.
pub fn program_for(context: usize, device_id: usize) -> Result<usize> {
    let cache = PROGS.get_or_init(|| std::sync::Mutex::new(vec![]));
    let mut list = cache.lock().unwrap();
    for item in &*list {
        if item.0 == context && item.1.program != 0 {
            return Ok(item.1.program);
        }
    }
    // Source string is a Rust &str; pass its raw byte buffer to OpenCL.
    let sb = KERNEL_SRC.as_bytes();
    let mut srcs = [sb.as_ptr() as *const std::ffi::c_void];
    let mut lens = [sb.len()];
    let mut e: i32 = 0;
    let pr = unsafe { clCreateProgramWithSource(context, 1, srcs.as_ptr(), lens.as_ptr(), &mut e) };
    if e != CL_SUCCESS || pr == 0 {
        return Err(err(e, "clCreateProgramWithSource"));
    }
    let prg = ProgGuard(pr);
    let bc = unsafe { clBuildProgram(pr, 1, &device_id, std::ptr::null(), std::ptr::null(), std::ptr::null()) };
    if bc != CL_SUCCESS {
        // prg drops here -> clReleaseProgram (no leak on build failure).
        return Err(err(bc, "clBuildProgram"));
    }
    // Hand the program to the cache (KernelProg owns it via its own Drop); the
    // transient guard must not release it again.
    let raw = prg.into_raw();
    list.push((context, KernelProg { program: raw }));
    Ok(raw)
}

fn create_kernel(program: usize, name: &str) -> Result<usize> {
    // clCreateKernel expects a NUL-terminated C string; a plain Rust &str carries
    // no terminator, so build one (this is what the docs/devloop call out).
    let name = match std::ffi::CString::new(name) {
        Ok(n) => n,
        Err(_) => return Err(Error::Msg(format!(
            "opencl create_kernel: kernel name {name:?} contains a NUL byte"
        ))),
    };
    let mut e: i32 = 0;
    let k = unsafe { clCreateKernel(program, name.as_ptr() as *const std::ffi::c_void, &mut e) };
    if e != CL_SUCCESS || k == 0 {
        return Err(err(e, "clCreateKernel"));
    }
    Ok(k)
}

/// Set one kernel argument.  `value` is the address (as usize) of the argument
/// bytes; OpenCL copies them synchronously, so the backing storage only needs
/// to be valid for the duration of this call.
fn set_arg(kernel: usize, index: u32, value_addr: usize, size: usize) -> Result<()> {
    let ptr = unsafe { value_addr as *const std::ffi::c_void };
    let rc = unsafe { clSetKernelArg(kernel, index, size, ptr) };
    if rc != CL_SUCCESS {
        return Err(err(rc, "clSetKernelArg"));
    }
    Ok(())
}

fn addr_of<T: Sized>(v: &T) -> usize {
    (v as *const T) as usize
}

fn run_nd(queue: usize, kernel: usize, gsz: &[usize], dim: u32) -> Result<()> {
    let rc = unsafe { clEnqueueNDRangeKernel(queue, kernel, dim, std::ptr::null(), gsz.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null_mut()) };
    if rc != CL_SUCCESS {
        return Err(err(rc, "clEnqueueNDRangeKernel"));
    }
    Ok(())
}

/// RAII guard that releases a `cl_program` on drop unless ownership is handed
/// off to the program cache (whose `KernelProg` carries its own Drop).
struct ProgGuard(usize);
impl ProgGuard {
    fn into_raw(self) -> usize {
        let v = self.0;
        std::mem::forget(self);
        v
    }
}
impl Drop for ProgGuard {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe { clReleaseProgram(self.0) };
        }
    }
}

/// RAII guard that releases a `cl_kernel` on drop — on BOTH success and error
/// paths, so a failure between `clCreateKernel` and the final enqueue no longer
/// leaks the kernel handle.
struct KernelGuard(usize);
impl KernelGuard {
    fn into_raw(self) -> usize {
        let v = self.0;
        std::mem::forget(self);
        v
    }
}
impl Drop for KernelGuard {
    fn drop(&mut self) {
        if self.0 != 0 {
            unsafe { clReleaseKernel(self.0) };
        }
    }
}

pub fn run_affine(ctx: usize, dev: usize, queue: usize, x: usize, out: usize, n: usize, mul: f32, add: f32) -> Result<()> {
    let program = program_for(ctx, dev)?;
    let k = KernelGuard(create_kernel(program, "kaffine")?);
    let (n32, b0, b1) = (n as i32, x, out);
    set_arg(k.0, 0, addr_of(&b0), 8)?;
    set_arg(k.0, 1, addr_of(&b1), 8)?;
    set_arg(k.0, 2, addr_of(&n32), 4)?;
    set_arg(k.0, 3, addr_of(&mul), 4)?;
    set_arg(k.0, 4, addr_of(&add), 4)?;
    run_nd(queue, k.0, &[n.max(1)], 1)?;
    // k drops here -> clReleaseKernel.
    Ok(())
}

pub fn run_unary(ctx: usize, dev: usize, queue: usize, op: &str, x: usize, out: usize, n: usize) -> Result<()> {
    let program = program_for(ctx, dev)?;
    let kname = match op {
        "exp" => "kexp",
        "log" => "klog",
        "sqrt" => "ksqrt",
        "sqr" => "ksqr",
        "neg" => "kneg",
        "recip" => "krecip",
        _ => "kexp",
    };
    let k = KernelGuard(create_kernel(program, kname)?);
    let (n32, b0, b1) = (n as i32, x, out);
    set_arg(k.0, 0, addr_of(&b0), 8)?;
    set_arg(k.0, 1, addr_of(&b1), 8)?;
    set_arg(k.0, 2, addr_of(&n32), 4)?;
    run_nd(queue, k.0, &[n.max(1)], 1)?;
    // k drops here -> clReleaseKernel.
    Ok(())
}

pub fn run_binary(ctx: usize, dev: usize, queue: usize, op: &str, a: usize, b: usize, out: usize, n: usize) -> Result<()> {
    let program = program_for(ctx, dev)?;
    let kname = match op {
        "add" => "kadd",
        "sub" => "ksub",
        "mul" => "kmul",
        _ => "kadd",
    };
    let k = KernelGuard(create_kernel(program, kname)?);
    let (n32, b0, b1, b2) = (n as i32, a, b, out);
    set_arg(k.0, 0, addr_of(&b0), 8)?;
    set_arg(k.0, 1, addr_of(&b1), 8)?;
    set_arg(k.0, 2, addr_of(&b2), 8)?;
    set_arg(k.0, 3, addr_of(&n32), 4)?;
    run_nd(queue, k.0, &[n.max(1)], 1)?;
    // k drops here -> clReleaseKernel.
    Ok(())
}

/// Max rows per single GEMM submission. Splitting M into bands of this many rows
/// keeps every submission short enough that the amdgpu compute-ring-timeout
/// watchdog never declares it hung and hard-recovers the context (the crash
/// root cause). Empirically a 32-row flat band at H=4096 completes in ~50-90ms,
/// far below the ~450ms+ single dispatches that were marginal/flaky.
const M_BAND: usize = 32;

pub fn run_matmul(ctx: usize, dev: usize, queue: usize, a: usize, b: usize, out: usize, (m, n, _k): (usize, usize, usize)) -> Result<()> {
    let program = program_for(ctx, dev)?;
    let k = KernelGuard(create_kernel(program, "kmatmul")?);
    let (N, K, b0, b1, b2) = (n as i32, _k as i32, a, b, out);
    set_arg(k.0, 0, addr_of(&b0), 8)?;
    set_arg(k.0, 1, addr_of(&b1), 8)?;
    set_arg(k.0, 2, addr_of(&b2), 8)?;
    set_arg(k.0, 4, addr_of(&N), 4)?;
    set_arg(k.0, 5, addr_of(&K), 4)?;
    let mut row0 = 0usize;
    while row0 < m {
        let band_m = (m - row0).min(M_BAND);
        let (BM, R0) = (band_m as i32, row0 as i32);
        set_arg(k.0, 3, addr_of(&BM), 4)?;
        set_arg(k.0, 6, addr_of(&R0), 4)?;
        run_nd(queue, k.0, &[band_m.max(1), n.max(1)], 2)?;
        row0 += band_m;
    }
    // k drops here -> clReleaseKernel.
    Ok(())
}

/// Whether we should use native kernels at all (opt-in env gate).
pub fn native_enabled() -> bool {
    match std::env::var("JOSHUA_OPENCL_NATIVE") {
        Ok(s) => !s.is_empty() && s != "0",
        Err(_) => false,
    }
}

pub fn has_unary(name: &str) -> bool { matches!(name, "exp" | "log" | "sqrt" | "sqr" | "neg" | "recip") }
pub fn has_binary(name: &str) -> bool { matches!(name, "add" | "sub" | "mul") }

/// Count of native OpenCL kernels that actually executed (not silently fallen
/// back). Used by the parity test to assert that native compute really ran rather
/// than accepting a CPU-fallback result.
static NATIVE_EXEC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn note_native_exec() {
    NATIVE_EXEC.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub fn native_exec_count() -> usize {
    NATIVE_EXEC.load(std::sync::atomic::Ordering::Relaxed)
}
