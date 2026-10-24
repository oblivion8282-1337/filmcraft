//! Smart rendering of ProRes and DNx exports ([`ExportSettings::smart_render`]).
//!
//! Both codecs code every frame on its own, so an export in them can take any frame straight from
//! a source file: a frame that shows one stored picture of a source untouched
//! ([`filmcraft_render::passthrough`]) and is coded the way the export codes its frames goes into
//! the file as it is stored, and the encoder only sees the frames that changed. The copy is the
//! source's picture bit for bit, where encoding the rendered frame again would add a generation of
//! compression (and pass through 8-bit RGB on the way).
//!
//! "Coded the way the export codes" is strict, so the file stays one consistent stream that says
//! what it holds: the same ProRes profile (the sample entry's four-character code is the only place
//! it is stated) or DNx compression ID and depth, the output's frame size, the output's field
//! order (progressive, ProRes fields in the order asked for, or the interlaced DNxHD ID asked for:
//! a frame's fields are never copied into a progressive file, nor in the other order), no alpha, and Rec. 709 colour as the export signals it. The
//! output must be the sequence's picture as rendered: same size, no Export effects or overlays, no
//! HDR. Anything else is rendered and encoded as before.

use std::sync::Arc;

use filmcraft_frame::CodedFormat;
use filmcraft_media::CodedSample;
use filmcraft_project::{ItemId, Project};
use filmcraft_render::{RenderOptions, SourceProvider};
use filmcraft_time::Tick;

use crate::{ExportSettings, Format};

/// What a copied frame must be coded as.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Coding {
    ProRes { fourcc: [u8; 4], chroma: filmcraft_prores::ChromaFormat, interlace: filmcraft_prores::Interlace },
    Dnx { cid: u32, bit_depth: u8, interlaced: bool },
}

/// The smart rendering of one export: which stored pictures it may copy.
#[derive(Clone, Debug)]
pub(crate) struct SmartRender {
    coding: Coding,
    width: u32,
    height: u32,
}

/// What an export renders its frames as, for [`SmartRender::new`].
pub(crate) struct OutputPicture {
    pub width: u32,
    pub height: u32,
    /// The picture is the sequence's frame as rendered, at its size (no scaling, padding or crop).
    pub as_rendered: bool,
    pub hdr: bool,
}

impl SmartRender {
    /// The smart rendering of an export with `settings` to `out`, when it can copy anything.
    pub fn new(settings: &ExportSettings, out: &OutputPicture) -> Option<SmartRender> {
        if !settings.smart_render || !out.as_rendered || out.hdr || settings.keeps_alpha() || settings.effects.any_video() {
            return None;
        }
        let coding = match settings.video_format() {
            Format::ProRes => {
                let p = crate::prores_profile(&settings.prores_profile);
                Coding::ProRes { fourcc: p.fourcc(), chroma: p.chroma(), interlace: settings.prores_interlace() }
            }
            Format::DnxHr => {
                let c = crate::dnx_config(&settings.dnx_profile, out.width, out.height);
                // DNx 4:4:4 is left out: other encoders code it as RGB, ours as Y'CbCr, and the
                // header does not say which a frame holds
                if c.chroma() != filmcraft_dnx::ChromaFormat::Yuv422 || c.alpha.is_some() {
                    return None;
                }
                Coding::Dnx { cid: c.cid(), bit_depth: c.bit_depth, interlaced: c.hd_cid.is_some_and(crate::dnx_interlaced) }
            }
            _ => return None,
        };
        Some(SmartRender { coding, width: out.width, height: out.height })
    }

    /// The stored picture to copy for the output frame at timeline `t` of `seq`, if there is one
    /// that fits.
    pub fn stored(&self, project: &Project, seq: ItemId, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> Option<Arc<[u8]>> {
        let s = filmcraft_render::passthrough::passthrough_frame(project, seq, t, opts, sources)?;
        // the colour the export signals (`ColorSignal::default`, BT.709) is the picture's
        if s.color != filmcraft_color::ColorInfo::REC709 {
            return None;
        }
        self.fits(&s).then_some(s.data)
    }

    /// Whether a stored picture is coded as this export codes its frames. Its structure is checked
    /// as the decoders check it, so a picture whose headers do not parse is rendered (and shows
    /// what the decoder makes of it) rather than copied.
    fn fits(&self, s: &CodedSample) -> bool {
        match self.coding {
            Coding::ProRes { fourcc, chroma, interlace } => {
                if s.format != CodedFormat::ProRes || s.fourcc != fourcc {
                    return false;
                }
                let Ok(l) = filmcraft_prores::frame_layout(&s.data) else { return false };
                let h = l.header;
                (u32::from(h.width), u32::from(h.height)) == (self.width, self.height)
                    && h.chroma == chroma
                    && h.interlace == interlace
                    && h.alpha == filmcraft_prores::AlphaType::None
                    // the frame's own colour fields: BT.709 or unspecified (decoders that read
                    // them, rather than the container's, show it as our frames)
                    && [h.color.primaries, h.color.transfer, h.color.matrix].iter().all(|c| matches!(c, 0..=2))
            }
            Coding::Dnx { cid, bit_depth, interlaced } => {
                if s.format != CodedFormat::Dnx {
                    return false;
                }
                let Ok(l) = filmcraft_dnx::frame_layout(&s.data) else { return false };
                let h = &l.header;
                (l.width, l.height) == (self.width, self.height)
                    && h.cid == cid
                    && h.bit_depth == bit_depth
                    && h.interlaced == interlaced
                    && h.chroma == filmcraft_dnx::ChromaFormat::Yuv422
                    && !h.rgb_planes()
                    && !h.alpha
                    && matches!(h.color_volume, filmcraft_dnx::ColorVolume::Bt709 | filmcraft_dnx::ColorVolume::OutOfBand)
            }
        }
    }
}
