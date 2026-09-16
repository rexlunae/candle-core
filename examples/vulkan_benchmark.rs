//! Vulkan backend micro-benchmark: measures native kernel throughput on a
//! Vulkan device (bring-up target: AMD Renoir iGPU / RADV).
//!
//! Requires the `vulkan` feature. Native kernels run only when
//! `JOSHUA_VULKAN_NATIVE=1`; without it the ops fall back to CPU, which this
//! benchmark then measures for comparison. Run on the Vulkan host with:
//!
//!   JOSHUA_VULKAN_NATIVE=1 cargo run --release --features vulkan \
//!       --example vulkan_benchmark -- <m> <k> <n> <iters>
//!
//! Defaults to a 1024x1024x1024 matmul (a size that runs reliably on iGPUs;
//! very large single-dispatches such as 4096^3 can trip a compute-shader
//! watchdog / device-lost) and a 1M-element affine, each over 50 iters.

use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().collect::<Vec<String>>();
    let m: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    // `iters` of 0 would make every timing/throughput figure a divide-by-zero,
    // so clamp it to a usable minimum.
    let iters: usize = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .map(|v| v.max(1))
        .unwrap_or(50);

    let native = std::env::var("JOSHUA_VULKAN_NATIVE").map(|v| v == "1").unwrap_or(false);
    let dev = match candle_core::VulkanDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping vulkan benchmark: {e}");
            return Ok(());
        }
    };
    let dev = Device::Vulkan(dev);
    println!("device: {:?}{}", dev, if native { " (native kernels ON)" } else { " (CPU fallback)" });

    // ---- Matmul: (m,k) @ (k,n) ----
    let a_v: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
    let b_v: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.25).collect();
    let a = Tensor::from_vec(a_v, (m, k), &dev)?;
    let b = Tensor::from_vec(b_v, (k, n), &dev)?;
    // Warm up once (first dispatch also builds the pipeline).
    let _ = a.matmul(&b)?;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let c = a.matmul(&b)?;
        candle_core::Device::synchronize(&dev)?;
        drop(c);
    }
    let el = start.elapsed().as_secs_f64() / iters as f64;
    let mm_flops = 2.0 * m as f64 * n as f64 * k as f64 / el;
    println!(
        "matmul {m}x{k}@{k}x{n}: {:.3} ms/iter, {:.2} GFLOPS",
        el * 1e3,
        mm_flops / 1e9
    );

    // ---- Affine (elementwise read-modify-write) ----
    let x_v: Vec<f32> = (0..1_048_576).map(|i| i as f32 * 0.5).collect();
    let x = Tensor::from_vec(x_v, (1_048_576,), &dev)?;
    let _ = x.affine(1.5, -0.25)?;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let y = x.affine(1.5, -0.25)?;
        candle_core::Device::synchronize(&dev)?;
        drop(y);
    }
    let el = start.elapsed().as_secs_f64() / iters as f64;
    println!(
        "affine 1M: {:.3} ms/iter, {:.2} G elem/s",
        el * 1e3,
        (1_048_576.0 / el) / 1e9
    );

    Ok(())
}