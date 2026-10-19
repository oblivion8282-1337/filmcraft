//! Camera-RAW frames that stay in GPU memory (ADR 0002), end to end on real hardware: a real
//! bridge decodes into exported CUDA memory, `vk_interop` imports it into a wgpu Vulkan device and
//! the GPU compositor copies it into a texture — the composite must be pixel-identical to the same
//! frame uploaded from host memory (its CPU fallback, decoded again over the host path). Also
//! times both paths. Needs a real clip and its vendor bridge: `FILMCRAFT_RAWCAM_REAL=<.braw/.R3D>
//! cargo test -p filmcraft-platform --test vk_interop -- --ignored --nocapture`.
#![cfg(target_os = "linux")]

use std::sync::Arc;
use std::time::Instant;

use filmcraft_frame::{PixelData, VideoFrame};
use filmcraft_geom::Affine;
use filmcraft_gpu::GpuCompositor;
use filmcraft_media::FrameRequest;
use filmcraft_render::Blend;
use filmcraft_render::plan::{FramePlan, PlanLayer};
use filmcraft_time::Tick;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

fn composite(c: &mut GpuCompositor, f: Arc<VideoFrame>) -> Vec<u8> {
    let (w, h) = (f.width as usize, f.height as usize);
    c.composite(&FramePlan::Layers { width: w, height: h, layers: vec![PlanLayer::new(f, Affine::IDENTITY, 1.0, Blend::Normal)] });
    c.read_output().map(|o| o.2).unwrap_or_default()
}

#[test]
#[ignore = "needs a real camera file (FILMCRAFT_RAWCAM_REAL), its vendor bridge and a CUDA GPU"]
fn real_gpu_frames_composite_like_host_frames() {
    let Some(path) = std::env::var_os("FILMCRAFT_RAWCAM_REAL").map(std::path::PathBuf::from) else {
        panic!("set FILMCRAFT_RAWCAM_REAL to a .braw/.R3D file");
    };
    let Some((dev, queue)) = device() else { panic!("no Vulkan device") };
    assert!(filmcraft_platform::vk_interop::register(&dev), "the device can import GPU frames");
    let src = match filmcraft_codecs::rawcam::path_opener(&path) {
        Some(Ok(s)) => s,
        other => panic!("open failed: {:?}", other.map(|r| r.err())),
    };
    let rate = src.info().frame_rate();
    let req = |k: i64, scale: f32| FrameRequest { time: Tick::from_rational(k, rate.den, rate.num), scale };
    let mut c = GpuCompositor::new(&dev, &queue);
    for (k, scale) in [(3, 1.0), (4, 0.5), (5, 0.25)] {
        let gpu = filmcraft_media::cancel::with_gpu_frames(true, || src.video_frame(req(k, scale))).unwrap_or_else(|e| panic!("frame {k}: {e:?}"));
        let PixelData::External(_) = &gpu.data else { panic!("frame {k} at {scale}: a GPU frame, got {}", gpu.format_label()) };
        let copies = c.external_copies();
        let via_gpu = composite(&mut c, gpu.clone());
        assert_eq!(c.external_copies(), copies + 1, "frame {k}: imported and copied on the GPU");
        let host = Arc::new(gpu.cpu_frame().into_owned());
        assert!(matches!(host.data, PixelData::Rgba8(_)));
        let via_host = composite(&mut c, host);
        assert_eq!(via_gpu.len(), via_host.len());
        let diff = via_gpu.iter().zip(&via_host).filter(|(a, b)| a != b).count();
        eprintln!("frame {k} at {scale} ({}×{}): {diff} differing bytes", gpu.width, gpu.height);
        assert_eq!(diff, 0, "frame {k}: GPU path and host path composite alike");
    }

    // Time decode + composite of 24 full-resolution frames per path (fresh frames each).
    // (frames the read-ahead decoded may come from the cache either way: counted, not asserted)
    let run = |c: &mut GpuCompositor, from: i64, gpu: bool| {
        let (t, mut external) = (Instant::now(), 0);
        for k in from..from + 24 {
            let f = filmcraft_media::cancel::with_gpu_frames(gpu, || src.video_frame(req(k, 1.0))).unwrap_or_else(|e| panic!("frame {k}: {e:?}"));
            external += usize::from(matches!(f.data, PixelData::External(_)));
            c.composite(&FramePlan::Layers { width: 1920, height: 1080, layers: vec![PlanLayer::new(f, Affine::scale(0.3, 0.3), 1.0, Blend::Normal)] });
            let _ = dev.poll(wgpu::PollType::wait_indefinitely());
        }
        (t.elapsed().as_secs_f64() * 1000.0 / 24.0, external)
    };
    let _ = run(&mut c, 30, true); // warm both paths
    let _ = run(&mut c, 140, false);
    let (gpu_ms, gpu_ext) = run(&mut c, 60, true);
    let (host_ms, host_ext) = run(&mut c, 180, false);
    eprintln!(
        "full resolution, decode + composite per frame: GPU path {gpu_ms:.1} ms ({gpu_ext}/24 GPU frames), host path {host_ms:.1} ms ({host_ext}/24 GPU frames)"
    );
    assert!(gpu_ext >= 20, "the GPU path decodes into GPU memory");
}
