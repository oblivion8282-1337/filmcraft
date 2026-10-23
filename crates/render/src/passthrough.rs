//! Smart rendering's question: is an output frame exactly one stored source picture?
//!
//! An export to the codec its media is already in (ProRes to ProRes, DNx to DNx) can copy such a
//! frame's coded bytes into its file instead of decoding, compositing and encoding it again: much
//! faster, and without the generation loss of another encode. [`passthrough_frame`] says when the
//! renderer would draw nothing but that picture, by the rules the CPU reference renderer follows:
//! the top-most visible clip covers the frame with an opaque picture placed as it is, and an opaque
//! layer at full opacity in Normal mode replaces whatever is under it (`render_seq_tracks`). The
//! export then decides whether the stored picture fits its output (codec, profile, size, colour).
//!
//! Nothing is decoded for the question, and no frame cache is touched: the stored picture's size,
//! alpha and colour come with it (`MediaSource::coded_sample`). A frame the export renders after
//! all therefore finds the caches as it would have without smart rendering.

use filmcraft_media::CodedSample;
use filmcraft_project::{ItemId, ItemKind, Project, Sequence, Track, TrackItem};
use filmcraft_time::Tick;

use crate::{Blend, MAX_NEST_DEPTH, RenderOptions, SourceProvider, colorman, motion_matrix, output_size};

/// The stored source picture the frame of `seq_id` at `t` shows untouched, if it is one: the
/// top-most visible clip at `t` is plain media (no standard effects, no masks) placed exactly over
/// the whole frame (an identity Motion at the sequence's size), fully opaque in Normal mode, with no
/// transition, no speed blending (Frame Blending, Optical Flow) and no colour management (Rec. 709
/// media in the plain Rec. 709 pipeline), its picture opaque with a stored form — or a nested
/// sequence or multi-camera clip shown that way whose own frame is such a picture. `None` for
/// anything else, including a reduced-size render and burnt-in captions over the frame.
pub fn passthrough_frame(project: &Project, seq_id: ItemId, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> Option<CodedSample> {
    let seq = project.sequence(seq_id)?;
    if opts.scale != 1.0 || !seq.settings.color.is_plain() {
        return None;
    }
    if opts.captions {
        let (w, h) = output_size(seq, 1.0);
        if !crate::caption_overlays(seq, t, w, h).is_empty() {
            return None;
        }
    }
    top(project, seq, t, opts, sources, 0)
}

/// What a track shows at a time.
enum Shown {
    Nothing,
    Stored(CodedSample),
    /// A picture that is not a stored one untouched (or not only one).
    Other,
}

/// The picture of the top-most track of `seq` that shows something at `t`.
fn top(project: &Project, seq: &Sequence, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider, nest: u32) -> Option<CodedSample> {
    for tr in seq.video_tracks.iter().rev().filter(|tr| tr.enabled) {
        match on_track(project, seq, tr, t, opts, sources, nest) {
            Shown::Nothing => continue,
            Shown::Stored(p) => return Some(p),
            Shown::Other => return None,
        }
    }
    // a black frame
    None
}

fn on_track(project: &Project, seq: &Sequence, tr: &Track, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider, nest: u32) -> Shown {
    if tr.transitions.iter().any(|x| x.range().contains(t)) {
        return Shown::Other;
    }
    let Some(item) = tr.item_at(t).filter(|i| i.enabled) else { return Shown::Nothing };
    if crate::is_adjustment(project, item) {
        // with effects off the renderer leaves an adjustment layer out
        return if opts.effects { Shown::Other } else { Shown::Nothing };
    }
    match item_picture(project, seq, item, t, opts, sources, nest) {
        Some(p) => Shown::Stored(p),
        None => Shown::Other,
    }
}

/// Whether Motion leaves a picture exactly where it is (the renderer's identity placement).
fn identity(m: &filmcraft_geom::Affine) -> bool {
    (m.a - 1.0).abs() < 1e-9 && (m.d - 1.0).abs() < 1e-9 && m.b == 0.0 && m.c == 0.0 && m.e.abs() < 1e-9 && m.f.abs() < 1e-9
}

fn item_picture(
    project: &Project,
    seq: &Sequence,
    item: &TrackItem,
    t: Tick,
    opts: RenderOptions,
    sources: &dyn SourceProvider,
    nest: u32,
) -> Option<CodedSample> {
    let mt = item.effect_time_at(t);
    if opts.effects && item.has_standard_effects() || item.has_opacity_masks() || crate::opacity_blend(item, mt) != (1.0, Blend::Normal) {
        return None;
    }
    let seq_size = (seq.settings.width, seq.settings.height);
    match &project.item(item.item)?.kind {
        ItemKind::Sequence(nested) => {
            let size = (nested.settings.width, nested.settings.height);
            if nest >= MAX_NEST_DEPTH || size != seq_size || nested.settings.color != seq.settings.color {
                return None;
            }
            if !identity(&motion_matrix(seq, item, size, Some(nested.settings.par), mt)) {
                return None;
            }
            let ft = item.source_time_at(t);
            match item.multicam_angle(nested) {
                // a multi-camera clip shows its angle's track alone, switched on or not
                Some(angle) => {
                    let tr = nested.angle_video_track_index(angle).and_then(|i| nested.video_tracks.get(i))?;
                    match on_track(project, nested, tr, ft, opts, sources, nest + 1) {
                        Shown::Stored(p) => Some(p),
                        Shown::Nothing | Shown::Other => None,
                    }
                }
                None => {
                    // a nested sequence's captions are part of its picture
                    if !crate::caption_overlays(nested, ft, size.0 as usize, size.1 as usize).is_empty() {
                        return None;
                    }
                    top(project, nested, ft, opts, sources, nest + 1)
                }
            }
        }
        ItemKind::Media(_) | ItemKind::Subclip { .. } => {
            let size = crate::source_size(project, item.item)?;
            if size != seq_size || !identity(&motion_matrix(seq, item, size, crate::source_par(project, item.item), mt)) {
                return None;
            }
            let src = sources.source(item.item)?;
            let rate = src.info().frame_rate();
            if crate::interpolation_blend(item, t, rate).is_some() {
                return None;
            }
            let sample = src.coded_sample(crate::video_source_time(item, t, rate))?;
            if sample.alpha || (sample.width, sample.height) != size {
                return None;
            }
            let cs = colorman::source_space_of(project, item.item, &sample.color);
            if colorman::needs_management_of(&seq.settings.color, cs, &sample.color) {
                return None;
            }
            Some(sample)
        }
        ItemKind::Graphic { .. } | ItemKind::AdjustmentLayer { .. } => None,
    }
}
