//! Pictures that live in GPU memory another process exported ([ADR 0002]): a camera-RAW bridge
//! decodes into device memory and hands FilmCraft the buffer instead of its pixels. The GPU
//! compositor imports the buffer and copies the picture into a texture without a host round trip;
//! every CPU consumer reads the pixels through the fallback the decoder supplies (it decodes the
//! frame again over the host path), so a CPU path never sees a missing picture, only a slower one.
//!
//! [ADR 0002]: https://github.com/storytold/filmcraft/blob/main/docs/adr/0002-gpu-frame-interop.md

use std::any::Any;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// One block of exported GPU memory (a bridge's frame buffer), imported at most once per GPU
/// device and kept for every frame that is decoded into it.
pub struct ExternalMemory {
    id: u64,
    size: u64,
    device_uuid: [u8; 16],
    handle: Box<dyn Any + Send + Sync>,
}

impl ExternalMemory {
    /// `id` is unique in the process (the importer caches imports by it), `size` the exported
    /// allocation size, `device_uuid` the GPU it lives on, `handle` what the importer needs to
    /// import it (an `OwnedFd` for a POSIX file-descriptor export).
    pub fn new(id: u64, size: u64, device_uuid: [u8; 16], handle: Box<dyn Any + Send + Sync>) -> Self {
        ExternalMemory { id, size, device_uuid, handle }
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn device_uuid(&self) -> [u8; 16] {
        self.device_uuid
    }
    /// The importer-specific handle (see [`ExternalMemory::new`]).
    pub fn handle(&self) -> &(dyn Any + Send + Sync) {
        self.handle.as_ref()
    }
}

impl std::fmt::Debug for ExternalMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalMemory").field("id", &self.id).field("size", &self.size).finish_non_exhaustive()
    }
}

/// The pixel layout of an external picture. Every layout holds *encoded* values (the frame's
/// transfer, or its media's colour space — log for camera RAW) and straight alpha; colour
/// management decodes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalLayout {
    /// 8 bits per channel.
    Rgba8,
    /// As `Rgba8`, blue first.
    Bgra8,
    /// Half floats (IEEE 754, little-endian) per channel, the encoded value normalised to 0..1: a
    /// 16-bit decode that is not cut to 8 bits.
    RgbaF16,
}

impl ExternalLayout {
    /// Bytes one pixel takes.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            ExternalLayout::Rgba8 | ExternalLayout::Bgra8 => 4,
            ExternalLayout::RgbaF16 => 8,
        }
    }
}

/// Supplies an external picture's pixels to the CPU in host memory (RGBA, 8 or 16 bits, tightly
/// packed rows, the same encoding), or `None` when they cannot be had (the picture then reads as
/// transparent black).
pub type CpuFallback = Box<dyn Fn() -> Option<crate::PixelData> + Send + Sync>;

/// The claim on an exported buffer a picture holds until the GPU has copied it: dropping it
/// gives the buffer back to the exporter.
pub struct GpuLease {
    pub memory: Arc<ExternalMemory>,
    _release: Option<Box<dyn Any + Send>>,
}

impl std::fmt::Debug for GpuLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLease").field("memory", &self.memory).finish_non_exhaustive()
    }
}

/// A picture in an [`ExternalMemory`]: rows of `pitch` bytes from `offset`.
///
/// The exporter's buffers are scarce (they are its VRAM): a picture gives its buffer back as soon
/// as the GPU compositor copied it into a texture of its own ([`ExternalImage::take_gpu`]), and a
/// picture that has neither its buffer nor fetched CPU pixels any more is *spent* — decoders
/// treat a cached spent picture as missing and decode it again.
pub struct ExternalImage {
    serial: u64,
    offset: u64,
    pitch: u32,
    layout: ExternalLayout,
    gpu: Mutex<Option<GpuLease>>,
    fetch: CpuFallback,
    cpu: OnceLock<crate::PixelData>,
}

impl ExternalImage {
    /// `release` is dropped when the buffer is given back: with the last reference to the
    /// picture, or once the GPU copied it ([`ExternalImage::take_gpu`]).
    pub fn new(memory: Arc<ExternalMemory>, offset: u64, pitch: u32, layout: ExternalLayout, fetch: CpuFallback, release: Option<Box<dyn Any + Send>>) -> Self {
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        ExternalImage {
            serial: SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            offset,
            pitch,
            layout,
            gpu: Mutex::new(Some(GpuLease { memory, _release: release })),
            fetch,
            cpu: OnceLock::new(),
        }
    }
    /// Unique in the process (pictures are never confused, even at a reused address).
    pub fn serial(&self) -> u64 {
        self.serial
    }
    /// The buffer, while the picture still holds it.
    pub fn memory(&self) -> Option<Arc<ExternalMemory>> {
        self.gpu.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(|l| l.memory.clone())
    }
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn pitch(&self) -> u32 {
        self.pitch
    }
    pub fn layout(&self) -> ExternalLayout {
        self.layout
    }

    /// Take the picture's buffer to copy it on the GPU: the caller drops the lease once the copy
    /// has run (the buffer then goes back to the exporter). Only one consumer gets it; `None`
    /// when it is gone (taken already, or the picture never had a usable one).
    pub fn take_gpu(&self) -> Option<GpuLease> {
        self.gpu.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    /// Whether the picture still holds its buffer.
    pub fn has_gpu(&self) -> bool {
        self.gpu.lock().unwrap_or_else(PoisonError::into_inner).is_some()
    }

    /// Neither the buffer nor CPU pixels any more: showing it again means decoding it again.
    pub fn is_spent(&self) -> bool {
        !self.has_gpu() && self.cpu.get().is_none()
    }

    /// What the picture holds now: its buffer rows while it has the buffer, else its fetched CPU
    /// pixels (`width`×`height` RGBA8), else nothing.
    pub fn byte_size(&self, width: u32, height: u32) -> usize {
        if self.has_gpu() {
            usize::try_from(u64::from(self.pitch).saturating_mul(u64::from(height))).unwrap_or(usize::MAX)
        } else {
            match self.cpu.get() {
                Some(crate::PixelData::Rgba16(d)) => d.len() * 2,
                Some(crate::PixelData::Rgba8(d)) => d.len(),
                Some(_) => (width as usize).saturating_mul(height as usize).saturating_mul(4),
                None => 0,
            }
        }
    }

    /// The picture's pixels in host memory (RGBA8 or RGBA16 rows of a `width`×`height` picture),
    /// fetched once through the fallback; transparent black when it fails or delivers anything
    /// else.
    pub fn cpu(&self, width: u32, height: u32) -> crate::PixelData {
        self.cpu
            .get_or_init(|| {
                let want = (width as usize).saturating_mul(height as usize).saturating_mul(4);
                let fetched = (self.fetch)();
                match fetched {
                    Some(crate::PixelData::Rgba8(d)) if d.len() == want => crate::PixelData::Rgba8(d),
                    Some(crate::PixelData::Rgba16(d)) if d.len() == want => crate::PixelData::Rgba16(d),
                    other => {
                        log::warn!(
                            "GPU frame {}×{height}: the CPU fallback {} — showing it transparent",
                            width,
                            if other.is_some() { "delivered the wrong size or layout" } else { "failed" }
                        );
                        crate::PixelData::Rgba8(Arc::new(vec![0; want]))
                    }
                }
            })
            .clone()
    }

    /// Whether the CPU pixels have been fetched already.
    pub fn has_cpu_pixels(&self) -> bool {
        self.cpu.get().is_some()
    }
}

impl std::fmt::Debug for ExternalImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalImage")
            .field("serial", &self.serial)
            .field("offset", &self.offset)
            .field("pitch", &self.pitch)
            .field("layout", &self.layout)
            .field("gpu", &self.has_gpu())
            .finish_non_exhaustive()
    }
}

/// The GPU an importer is registered for (its `deviceUUID`): decoders only produce external
/// pictures that can be imported there.
static IMPORT_DEVICE: Mutex<Option<[u8; 16]>> = Mutex::new(None);

/// Register (or, with `None`, withdraw) the GPU external pictures can be imported into.
pub fn set_import_device(uuid: Option<[u8; 16]>) {
    *IMPORT_DEVICE.lock().unwrap_or_else(PoisonError::into_inner) = uuid;
}

/// The GPU external pictures can be imported into, if any.
pub fn import_device() -> Option<[u8; 16]> {
    *IMPORT_DEVICE.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(fetch: CpuFallback) -> ExternalImage {
        ExternalImage::new(Arc::new(ExternalMemory::new(1, 4096, [0; 16], Box::new(()))), 0, 256, ExternalLayout::Rgba8, fetch, None)
    }

    #[test]
    fn the_cpu_fallback_runs_once_and_is_cached() {
        let calls = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let c = calls.clone();
        let img = image(Box::new(move || {
            c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(crate::PixelData::Rgba16(Arc::new(vec![7; 2 * 2 * 4])))
        }));
        let px = |d: crate::PixelData| match d {
            crate::PixelData::Rgba16(d) => d.as_ref().clone(),
            _ => vec![],
        };
        assert_eq!(px(img.cpu(2, 2)), vec![7; 16]);
        assert_eq!(px(img.cpu(2, 2)), vec![7; 16]);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn a_failing_or_short_fallback_reads_transparent() {
        let bytes = |d: crate::PixelData| match d {
            crate::PixelData::Rgba8(d) => d.as_ref().clone(),
            _ => vec![1],
        };
        assert_eq!(bytes(image(Box::new(|| None)).cpu(2, 1)), vec![0; 8]);
        assert_eq!(bytes(image(Box::new(|| Some(crate::PixelData::Rgba8(Arc::new(vec![9; 3]))))).cpu(2, 1)), vec![0; 8]);
    }

    #[test]
    fn the_release_guard_drops_with_the_picture() {
        struct Flag(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Flag {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let img = Arc::new(ExternalImage::new(
            Arc::new(ExternalMemory::new(1, 4096, [0; 16], Box::new(()))),
            0,
            256,
            ExternalLayout::Rgba8,
            Box::new(|| None),
            Some(Box::new(Flag(released.clone()))),
        ));
        let other = img.clone();
        drop(img);
        assert!(!released.load(std::sync::atomic::Ordering::Relaxed), "still referenced");
        drop(other);
        assert!(released.load(std::sync::atomic::Ordering::Relaxed));
    }

    /// The GPU copy takes the buffer once and gives it back when the lease drops, while the
    /// picture lives on: then spent (nothing held, nothing counted) until CPU pixels are fetched.
    #[test]
    fn the_gpu_takes_the_buffer_once_and_gives_it_back() {
        struct Flag(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Flag {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let img = ExternalImage::new(
            Arc::new(ExternalMemory::new(1, 4096, [0; 16], Box::new(()))),
            0,
            256,
            ExternalLayout::Rgba8,
            Box::new(|| Some(crate::PixelData::Rgba8(Arc::new(vec![5; 2 * 3 * 4])))),
            Some(Box::new(Flag(released.clone()))),
        );
        assert_eq!((img.byte_size(2, 3), img.is_spent()), (768, false));
        let lease = img.take_gpu();
        assert!(lease.is_some() && img.take_gpu().is_none(), "one consumer gets the buffer");
        assert!(!released.load(std::sync::atomic::Ordering::Relaxed), "held until the copy ran");
        drop(lease);
        assert!(released.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!((img.byte_size(2, 3), img.is_spent(), img.memory().is_none()), (0, true, true));
        assert!(matches!(img.cpu(2, 3), crate::PixelData::Rgba8(d) if d.len() == 24));
        assert_eq!((img.byte_size(2, 3), img.is_spent()), (24, false));
    }
}
