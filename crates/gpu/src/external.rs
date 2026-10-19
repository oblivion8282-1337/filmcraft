//! Pictures in another process's GPU memory ([`PixelData::External`], ADR 0002): the exported
//! buffer is imported into this device once and every picture in it is copied into a texture on
//! the GPU — no host round trip. Importing needs the graphics API underneath wgpu (`unsafe`), so
//! the platform layer implements [`ExternalImporter`] and registers it; without one (or when an
//! import fails) the picture is uploaded from its CPU fallback like any RGBA8 frame.
//!
//! [`PixelData::External`]: filmcraft_frame::PixelData::External

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use filmcraft_frame::{ExternalImage, ExternalLayout, ExternalMemory};

/// Imports exported GPU memory into a wgpu device.
pub trait ExternalImporter: Send + Sync {
    /// `memory` as a buffer of `device` usable as a copy source, or `None` when it cannot be
    /// imported there (another GPU, another backend, a driver refusal).
    fn import(&self, device: &wgpu::Device, memory: &ExternalMemory) -> Option<wgpu::Buffer>;
}

static IMPORTER: Mutex<Option<Arc<dyn ExternalImporter>>> = Mutex::new(None);

/// Register the process's importer (the platform layer does at start-up).
pub fn register_importer(importer: Arc<dyn ExternalImporter>) {
    *IMPORTER.lock().unwrap_or_else(PoisonError::into_inner) = Some(importer);
}

fn importer() -> Option<Arc<dyn ExternalImporter>> {
    IMPORTER.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// The texture format an external layout is sampled as (8-bit layouts sRGB-encoded like `Rgba8`
/// frames, half floats as they are), and the compositor's source kind for it (0: RGBA, 1:
/// premultiplied linear float).
fn texture_format(layout: ExternalLayout) -> (wgpu::TextureFormat, u32) {
    match layout {
        ExternalLayout::Rgba8 => (wgpu::TextureFormat::Rgba8UnormSrgb, 0),
        ExternalLayout::Bgra8 => (wgpu::TextureFormat::Bgra8UnormSrgb, 0),
        ExternalLayout::RgbaF16 => (wgpu::TextureFormat::Rgba16Float, 1),
    }
}

/// Whether a `width`×`height` picture of `bpp`-byte pixels at `offset` with rows `pitch` apart
/// lies inside `size` bytes and meets wgpu's copy rules (rows a multiple of 256 bytes, at least a
/// row of pixels).
pub(crate) fn layout_fits(width: u32, height: u32, bpp: u32, offset: u64, pitch: u32, size: u64) -> bool {
    let row = u64::from(width) * u64::from(bpp);
    if width == 0 || height == 0 || !pitch.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) || u64::from(pitch) < row || !offset.is_multiple_of(4) {
        return false;
    }
    let end = u64::from(pitch).checked_mul(u64::from(height - 1)).and_then(|b| b.checked_add(row)).and_then(|b| b.checked_add(offset));
    end.is_some_and(|end| end <= size)
}

/// The buffers this device imported, by memory id; an entry goes once its memory is gone (no
/// frame and no decoder holds it any more).
#[derive(Default)]
pub(crate) struct Imports {
    buffers: HashMap<u64, (Weak<ExternalMemory>, wgpu::Buffer)>,
    /// Pictures copied on the GPU (stats).
    pub(crate) copies: u64,
    /// External pictures uploaded from their CPU fallback instead (stats).
    pub(crate) fallbacks: u64,
}

impl Imports {
    /// The texture holding `img` (a `width`×`height` picture), filled by a GPU copy from its
    /// imported buffer. The picture's buffer is taken for the copy and given back to the exporter
    /// once the copy has run on the GPU (the texture keeps the picture). `None` when the picture
    /// no longer holds its buffer, there is no importer, or the import or layout is unusable — the
    /// caller then uploads the CPU fallback.
    pub(crate) fn texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &Arc<ExternalImage>,
        width: u32,
        height: u32,
    ) -> Option<(wgpu::TextureView, u32)> {
        let lease = img.take_gpu()?;
        let memory = &lease.memory;
        if !layout_fits(width, height, img.layout().bytes_per_pixel(), img.offset(), img.pitch(), memory.size()) {
            log::warn!("GPU frame {width}×{height} (pitch {}, offset {}) does not fit its {}-byte buffer", img.pitch(), img.offset(), memory.size());
            return None;
        }
        self.buffers.retain(|_, (m, _)| m.strong_count() > 0);
        let buffer = match self.buffers.get(&memory.id()) {
            Some((_, b)) => b.clone(),
            None => {
                let b = importer()?.import(device, memory)?;
                self.buffers.insert(memory.id(), (Arc::downgrade(memory), b.clone()));
                b
            }
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("filmcraft-external"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format(img.layout()).0,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("filmcraft-external-copy") });
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: img.offset(), bytes_per_row: Some(img.pitch()), rows_per_image: Some(height) },
            },
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        // The exporter may reuse the buffer once the lease is dropped: only after the copy has
        // run on the GPU, not when it was submitted.
        queue.on_submitted_work_done(move || drop(lease));
        self.copies += 1;
        Some((texture.create_view(&Default::default()), texture_format(img.layout()).1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layout_must_lie_inside_its_buffer_and_keep_wgpus_row_alignment() {
        // 6048 px: a 24192-byte row, padded to 24320
        assert!(layout_fits(6048, 2556, 4, 0, 24320, 24320 * 2556));
        assert!(!layout_fits(6048, 2556, 4, 0, 24192, 24192 * 2556), "rows must be 256-byte aligned");
        assert!(layout_fits(6048, 2556, 4, 0, 24320, 24320 * 2555 + 24192), "the last row needs only its pixels");
        assert!(!layout_fits(6048, 2556, 4, 0, 24320, 24320 * 2555 + 24191), "short buffer");
        assert!(!layout_fits(64, 2, 4, 0, 128, 1 << 20), "pitch below a row of pixels");
        assert!(!layout_fits(0, 2, 4, 0, 256, 1 << 20));
        assert!(!layout_fits(64, 2, 4, u64::MAX - 8, 256, u64::MAX), "overflowing end");
        assert!(layout_fits(64, 2, 4, 512, 256, 512 + 256 + 256));
        // half floats: 8 bytes a pixel
        assert!(layout_fits(32, 2, 8, 0, 256, 512));
        assert!(!layout_fits(33, 2, 8, 0, 256, 1 << 20), "a 264-byte row in 256-byte rows");
    }
}
