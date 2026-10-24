# ADR 0004: intra-only codecs encode on the GPU (ProRes, DNxHD / DNxHR)

- **Status:** accepted (2026-10-24, requested and approved by the project owner); Step 0 done
  (`a686eee3`), Steps 1–2 open
- **Extends:** [ADR 0003](0003-gpu-intra-decode.md) — the same codecs, the other direction; the
  same rules: WGSL compute in `filmcraft-gpu`, the CPU codec stays the reference and the fallback,
  never a crash, never a wrong file

## Context

ProRes and DNx exports spend their time in the encoder. Measured through the running app
(2026-10-24, RTX 4090, Ryzen 7 7800X3D, 100 frames of 3840×2160 25p real camera footage, smart
rendering off, `perf.stats` → `export.stages`):

| Export | Render (overlapped) | Convert RGB → Y'CbCr | Encode (serial, incl. convert) | Wall |
|---|---:|---:|---:|---:|
| ProRes 422 HQ, CPU rendering | 2.9 s | 1.4 s | 9.5 s | 10.3 s |
| ProRes 422 HQ, GPU rendering | 2.6 s | 0.6 s | 7.9 s | 8.8 s |
| DNxHR HQ, CPU rendering | 3.1 s | – | 5.6 s | 6.5 s |
| DNxHR HQ, GPU rendering | 2.3 s | – | 3.8 s | 4.7 s |

The encoder is the bottleneck (`waitMs` 0: the encode side never waits for the renderer), and GPU
rendering ([ADR 0003, Export](0003-gpu-intra-decode.md#export-2026-10-20-later)) can only shave
the render side. Both codecs are built for parallel work, which is why ADR 0003 could decode them
on the GPU: ProRes slices (~4000 per UHD frame) are coded independently; a DNx scan line (~135
per UHD frame) is one entropy run, but **when encoding** every coefficient and every DC predictor
is known before a single bit is written, so even DNx's bit layout can be computed in parallel.
No vendor hardware encodes either codec outside Apple silicon (NVENC / AMF / Quick Sync do H.264,
HEVC, AV1 only), so the GPU's compute cores are the only portable accelerator.

Two findings from the same measurement session shape the decision:

1. **The encoders only see 8 bits.** For SDR exports `EncoderFrame` carries RGBA8
   (`crates/export/src/job.rs`, `EncoderFrame { rgba, hdr: None }`); the 10/12-bit ProRes / DNx
   encoders upconvert it. A 10-bit ramp with 64 luma levels comes out of a ProRes 422 HQ export
   with 19. Any GPU encoder must take ≥ 16-bit input, and the CPU path needs the same fix
   (step 0 below).
2. **The CPU encoders still had serial passes** (the DNx rate control's refinement: HQX 5.9 →
   14.9 fps once parallel). The CPU numbers above are therefore the baseline *after* that fix is
   committed, not before.

## Decision

FilmCraft gets GPU encoders for ProRes and DNx in WGSL compute shaders, as a second
implementation of the same bitstream writers, chosen per export and per frame, with the CPU
encoders as reference and fallback.

### Step 0 (prerequisite, CPU only): a 16-bit hand-off

`EncoderFrame` gains a ≥ 16-bit picture for SDR exports (16-bit or f32 R'G'B'A after the output
transfer function, the representation the HDR path already uses), and the ProRes / DNx / APV
encoders convert from it. RGBA8 stays for encoders that want 8 bits (H.264 8-bit, MJPEG, GIF…).
Regression test: a 10-bit ramp keeps its levels through a ProRes HQ and a DNxHR HQX export.
This is a bug fix in its own right and lands first, independent of the GPU work.

**Done 2026-10-24 (`a686eee3`):** ProRes, DNx at 10/12 bits, APV, DPX and OpenEXR get a 16-bit
picture, 8-bit outputs keep RGBA8, GPU rendering feeds the same path. The ramp keeps 64 of 64
levels (was 19) through ProRes 422 HQ and DNxHR HQX; H.264 output is unchanged. A 4K export now
keeps 4 frames per batch in flight instead of 8.

### Step 1: GPU encoder for frames handed over as planes

- **`filmcraft-gpu::intra_enc`** (L3, no `unsafe`): takes Y'CbCr(A) planes (u16, the encoder's
  bit depth) and an encoder configuration (`filmcraft_prores::EncoderConfig` /
  `filmcraft_dnx::EncoderConfig`), uploads the planes, runs the passes below and reads back only
  the coded frame (~1–4 MB for UHD instead of ~33 MB of planes).
- **Wiring** mirrors the GPU frame renderer: the apps (`filmcraft`, `filmcraft-cli`; L6) register
  an `export::register_encoder` factory that wraps the GPU encoder in `export::VideoEncoder`, as
  they register `GpuFrameRenderer` today. No new layer edge.
- **Selection:** the existing *Hardware Encoding* export setting (`hardware_encoding`: off /
  auto) also governs the GPU ProRes / DNx encoders; `auto` uses them when a compute-capable device
  exists. (Builder's call to rename it in the UI, e.g. "Hardware / GPU encoding".) `perf.stats`
  `export.hardware` counts GPU-encoded frames, sessions and declines, as for NVENC.
- **ProRes passes** (one dispatch each, all slices in parallel):
  1. `fdct`: 8×8 forward DCT per block into a coefficient buffer, plus per-slice data the rate
     control needs.
  2. `bits(q)`: per slice, quantise at `q` and count the coded bits (the CPU's `slice_bits`).
  3. Rate control stays the CPU's algorithm (`choose_quantisers`: smallest uniform `q` that fits
     the frame budget by bisection over `Q_MIN..=Q_MAX`, then one step finer on as many slices
     as the leftover allows, then the 16-bit size-field guard). The bisection's ~8 steps run on
     the GPU without a CPU round trip: a reduction sums the slice bits, a one-invocation `pick`
     pass chooses the next `q` and writes it for the next `bits` dispatch (dispatches recorded
     up front). The greedy refinement is a serial walk over ~4000 slices in a fixed order —
     cheap; it runs as one invocation, or on the CPU after reading back the per-slice bit counts
     (~16 KB), whichever measures faster.
  4. `write`: every slice entropy-codes its components at its final `q` into its own region of
     the output (sizes known from step 3, exclusive prefix sum → byte offsets), then the slice
     table and frame / picture headers are written (on the GPU, or by the CPU around the slice
     data; builder's call).
  - Alpha (4444): the alpha coding per slice is a serial run over the slice's samples — one
    invocation per slice, like the CPU.
  - Interlaced (TFF / BFF), once the CPU encoder supports it (in progress on
    `feat/gpu-prores-dnx`): two pictures per frame, each a field; same passes per picture.
- **DNx passes:** `fdct` per block; per-macroblock bit counts at `q` (DC differences known from
  the transformed DC values, as the CPU's `prepare` does); rate control as in the CPU encoder
  (uniform `q` by bisection, then per-macroblock refinement: the refinement order is a fixed
  strided walk, kept serial and identical to the CPU's); then the **parallel bit layout**: an
  exclusive prefix sum of macroblock bit lengths per scan line gives every macroblock its bit
  offset, each macroblock codes into a scratch buffer, and one invocation per 32-bit output word
  gathers the bits of the macroblocks that overlap it. Scan-line sizes are then padded and the
  scan index table written as the CPU does. Scope as the CPU encoder: DNxHR LB…444, 8/10/12
  bits, DNxHD CIDs including field-coded 1080i, RGB 4:4:4; alpha only if/when export offers it.

### Step 2: the frame stays on the GPU

With GPU rendering and a GPU encoder, the rendered linear-light frame is read back today only to
be converted and re-uploaded. Step 2 moves the export's post-render stages that run per pixel
(placement / scaling, image / name / timecode overlays, video limiter, output transfer function,
R'G'B' → Y'CbCr with the export's matrix, chroma siting and subsampling) into compute or render
passes on the same device, and feeds the encoder's input textures directly; only the coded frame
crosses the bus. Any setting the GPU path does not implement keeps the read-back path (Step 1)
— decided per export up front, never mid-file. Smart rendering is unchanged: copied frames never
reach any encoder.

### Correctness, fallbacks, never a crash

- **Reference:** the CPU encoder. The GPU encoder follows its arithmetic (same DCT basis, same
  quantisation rounding, same rate control), but bit-identical output is *not* required: a fused
  multiply-add can move a coefficient at a rounding tie and with it a slice's size. Required:
  every GPU-encoded frame parses with `filmcraft_prores::frame_layout` /
  `filmcraft_dnx::frame_layout`, decodes with our CPU decoder and with ffmpeg without error,
  stays within the profile's size limits, and its decoded picture is within 0.1 dB PSNR of the
  CPU encoder's at the same settings. (Builder's call: if an integer DCT in both encoders makes
  them bit-identical at no quality cost, prefer that — it makes every test an equality.)
- **Per-frame fallback:** before a GPU-coded frame goes to the muxer, the CPU parses its layout
  (cheap; ADR 0003 already has these parsers). A frame that does not parse, exceeds a size limit,
  or comes from a device that reported an error is encoded again on the CPU. Every frame of these
  codecs is intra, so a file may freely mix GPU- and CPU-encoded frames.
- **Up-front fallback:** no compute support, a frame above the device's texture or storage
  limits, a pipeline that fails validation (created inside an error scope), or an unsupported
  configuration: the CPU encoder takes the export, and `export.hardware.declined` counts it.
- **Limits:** coefficient and output buffers are sized from the frame and checked against
  `max_storage_buffer_binding_size`; larger frames are encoded in slice-row batches like the
  decoder's.
- **Throughput:** the export pipeline keeps one frame in flight on the GPU while the CPU muxes the
  previous one (double-buffered output), so the GPU encoder does not serialise on read-back.

## Goals and non-goals

- **Goal:** UHD ProRes 422 HQ and DNxHR HQ / HQX exports become render-bound instead of
  encode-bound. Targets on the reference machine, 100 UHD frames as above: GPU encode ≤ 10 ms
  per frame (ProRes HQ CPU: ~79 ms), Step 1 wall time ≤ 4 s (8.8 s today), Step 2 ≤ 2.5 s.
  The targets are for the review of the result, not promises; the measurement goes into
  `docs/performance.md` either way.
- **Goal:** identical feature coverage to the CPU encoders for everything the export offers
  (profiles, bit depths, sizes incl. odd and DCI, 4444 + alpha, interlaced where supported).
- **Non-goal:** VideoToolbox's ProRes engines on Apple silicon (`crates/platform`, ADR 0001) —
  a possible later addition behind the same factory; the compute path must work everywhere first.
- **Non-goal:** H.264 / HEVC / AV1 on compute shaders (hardware encoders exist for those).
- **Non-goal:** encoding for anything but export (proxy creation may reuse it later).
- **Anti-goals** (fail even if it "works"): files that Apple / Avid / ffmpeg decoders reject or
  mis-display; worse quality than the CPU encoder at the same profile; a GPU fault that crashes,
  hangs or silently corrupts an export; making CPU-only exports slower.

## Testing

- `crates/gpu/src/intra_enc_tests.rs`: for our test pictures and ffmpeg-decoded real frames,
  every profile / bit depth / chroma / alpha / interlace combination, odd sizes, 1 to 8
  macroblocks per slice: GPU frame parses, decodes on CPU and GPU (ADR 0003) and in ffmpeg, PSNR
  within 0.1 dB of the CPU encode, sizes within limits.
- Fault injection: a forced device error and a corrupted output buffer must produce a CPU-encoded
  frame, not a broken file.
- `tests/export_pipeline.rs`: exports with the GPU encoder on and off produce files of equal
  frame count, valid containers (MOV / MXF), and pictures within tolerance.
- Real media: the test library on the Extern drive (`FilmCraft-Testmaterial`, 55 ProRes / DNx
  variants cut from camera footage) and the Komodo originals, as for ADR 0003.
- Benchmarks before / after in `docs/performance.md` (medians, the GPU clock-burst caveat).

## Order of work

0. 16-bit hand-off (CPU), with its regression test.
1. ProRes 422 GPU encoder (Step 1), then 4444 + alpha, then interlaced.
2. DNx GPU encoder (Step 1): DNxHR, then DNxHD CIDs and field coding, then 4:4:4 / RGB.
3. GPU-resident export path (Step 2).
4. Tests on AMD / Intel / Apple GPUs (needs the project owner's machines).

## Consequences

- Two encoders per codec to keep in step; every change to a CPU encoder's rate control or
  bitstream writer needs the matching shader change, enforced by the comparison tests.
- `filmcraft-gpu` grows by an encoder of roughly the decoder's size (~1500 lines of WGSL and
  Rust per codec, judging by `prores.wgsl` / `dnx.wgsl` and `intra.rs`).
- The GPU stays busy during export with work that competed for the CPU before; on a machine
  with a weak GPU, `auto` may be slower than the CPU — the encoder measures its first frames and
  may hand the export to the CPU (builder's call whether this is worth it in Step 1).
