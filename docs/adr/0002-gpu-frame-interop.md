# ADR 0002: camera-RAW frames stay on the GPU (CUDA → Vulkan interop)

- **Status:** accepted (2026-10-19, requested by the project owner)
- **Extends:** [ADR 0001](0001-platform-ffi.md) — one more `#[allow(unsafe_code)]` module in
  `crates/platform`, same containment rules

## Context

Camera-RAW clips (`.braw`, `.R3D`) are decoded by external bridge processes that wrap the vendor
SDKs (`crates/codecs/src/rawcam`). The SDKs decode on the GPU, and until now every frame then went
GPU → host (the bridge's readback) → shared memory → a `Vec` in FilmCraft (`pread`) → the GPU again
(`write_texture`). For a 6K frame (62–75 MB) that host round trip cost about as much as the
decode itself: a full-resolution R3D seek took 22 ms, of which 10.7 ms were the decode and 11.3 ms
moving and swizzling pixels on the CPU.

A spike (Linux, RTX 4090, wgpu 30) showed that memory a CUDA process allocates with
`cuMemCreate` and exports as an opaque file descriptor can be imported into another process's
wgpu Vulkan device as a `wgpu::Buffer`: pixel-exact, import 0.1–0.3 ms once per buffer, and a
GPU-local buffer → texture copy of ~1 ms for 62 MB. `VK_KHR_external_memory_fd` is already enabled
by wgpu-hal whenever the driver has it.

## Decision

Bridges that can export GPU memory hand FilmCraft frames *as GPU buffers*; FilmCraft imports each
buffer once into its wgpu device and copies the frame into a texture on the GPU. The host path
stays, unchanged, as the fallback for everything else.

### Bridge protocol (multi-clip mode, capability `gpu`)

- **Opt-in:** `--persistent --multi --gpu-export <socket>`. FilmCraft listens on a Unix stream
  socket in a private (0700) directory and passes its path; the bridge connects once at start-up.
  If the bridge cannot connect, or CUDA export is unavailable (CPU decoding, no VMM support), it
  carries on without the capability.
- **Ready event:** with the capability, the `ready` event lists `gpu` in `caps` and adds
  `"gpu_uuid":"<32 lowercase hex digits>"` (`cuDeviceGetUuid` of the decoding device).
- **Request:** `frame <handle> <index> <scale> gpu` asks for a GPU frame (the scale word is then
  mandatory). Without `gpu` the frame comes as before (shared-memory slot or stdout).
- **Buffers:** the bridge allocates exportable device memory (`cuMemCreate`, POSIX fd handles,
  granularity-rounded) and pools it by size. Every buffer has an id (a decimal number, never
  reused within one process). The *first* time a buffer is used, the bridge sends its fd over the
  socket with `SCM_RIGHTS`, with a 16-byte payload: the id and the exported size (two
  little-endian `u64`), **before** it writes the frame event that names the buffer.
- **Reply:** `{"type":"frame","handle":H,"index":N,"width":W,"height":H,"format":"rgba8"|"bgra8","gpu":B,"pitch":P}`
  and the usual `verify` event; no pixels on stdout. Rows are `P` bytes apart (`P` a multiple of
  256, as wgpu requires for buffer → texture copies), starting at offset 0. All GPU work writing
  the buffer has completed (stream synchronised) before the event is written.
- **16 bits (`f16`):** a bridge whose decode has more than 8 bits (R3D: REDCuda's 16-bit RGB;
  BRAW: the SDK's `RGBAU16` output of its 12-bit raw)
  lists `f16` in the `ready` caps and answers `frame <h> <i> <scale> gpu f16` with
  `"format":"rgbaf16"`: four IEEE 754 half floats per pixel (8 bytes, little-endian), linear
  light — the sRGB EOTF applied to the very values the 8-bit path rounds — premultiplied by
  alpha. That is the compositor's working precision (`Rgba16Float`), so the decode reaches it
  uncut. FilmCraft asks for it whenever the bridge offers it (`FILMCRAFT_RAWCAM_F16=0` keeps 8
  bits); the CPU fallback of such a frame still reads 8-bit pixels.
- **Ownership:** a buffer named in a reply is the client's until it sends `release-gpu <B>`;
  the bridge never writes it meanwhile. A frame the bridge cannot (or will not) export — no
  buffer free under its export budget, an allocation failure — comes the old way; the client
  must accept both.
- **Freeing:** when the bridge frees a pooled buffer (idle trim, budget), it writes
  `{"type":"gpu-freed","gpu":B}`; the client drops its fd and imports of `B` once no frame uses
  it. Memory the client has imported stays valid until the client frees it (the driver
  reference-counts it).

### FilmCraft

- `filmcraft-frame` gets `PixelData::External(Arc<ExternalImage>)`: size, pitch, RGBA/BGRA, the
  shared memory (an opaque, importer-specific handle — the fd, kept in an `OwnedFd`), a release
  guard (dropped with the last reference: `release-gpu`), and a **CPU fallback**: the first CPU
  consumer gets the pixels through a closure the decoder supplies (the RAW source decodes the
  frame again over the host path) and they are cached. Every CPU code path resolves an external
  frame this way, so nothing can see a missing or wrong picture — only a slower one.
- **Who gets GPU frames:** a thread-local hint, `filmcraft_media::cancel::with_gpu_frames`, set by
  the monitor's frame jobs when they composite on the GPU. Layers the plan renders on the CPU are
  fetched without it. The RAW source asks for `gpu` only under the hint, only when an importer is
  registered for the bridge's GPU (UUID match), and never for thumbnails or background work.
- **Import:** `filmcraft-gpu` (L3, no `unsafe`) defines an importer trait and a process-wide
  registration; its upload path turns an external frame into a texture with one
  `copy_buffer_to_texture`, caching the imported `wgpu::Buffer` per buffer id. Without an
  importer, or when an import fails, it uploads the CPU fallback instead.
- **`crates/platform::vk_interop`** (Linux, `#[allow(unsafe_code)]`) implements the importer:
  `vkCreateBuffer` with `ExternalMemoryBufferCreateInfo(OPAQUE_FD)`, `vkAllocateMemory` with
  `ImportMemoryFdInfoKHR` (exact exported size, a `DEVICE_LOCAL` type), wrapped with
  `hal::vulkan::Buffer::from_raw_externally_owned` and `Device::create_buffer_from_hal`. It reads
  the adapter's `deviceUUID` (`VkPhysicalDeviceIDProperties`) for the UUID match. The app
  registers it at start-up.
- **Buffers go back as soon as they are copied.** The picture holds its buffer as a lease the
  compositor takes for its copy and drops once the copy has run on the GPU; the texture keeps the
  picture. Holding buffers for as long as some cache (the RAW cache, the monitor's plan cache,
  the compositor's upload cache) held the frame filled the bridge's export budget within seconds
  of playback, and the rest came over the host. A cached picture whose buffer went back is
  *spent*: the source decodes it again when it is asked for once more. While it holds its
  buffer, an external frame counts its bytes in the camera-RAW cache budget like a host frame.
- The Source monitor composites media on the GPU as well (`plan_item`, `Target::ItemPlan`), with
  a compositor and texture of its own, so its frames take the same path.
- `perf.stats` reports `decode.rawcam` (GPU frames, host frames although the GPU was wanted,
  other host frames, CPU fallbacks) and `gpu.externalCopies` / `gpu.externalFallbacks`.

## Consequences

- Linux + NVIDIA + Vulkan first. Windows is the same design with Win32 handles
  (`CU_MEM_HANDLE_TYPE_WIN32`, `VK_KHR_external_memory_win32`, `DuplicateHandle` instead of
  `SCM_RIGHTS`); macOS has no CUDA and keeps the host path.
- Buffers in flight live in the bridge's VRAM; they go back after the GPU copy (measured: the
  BRAW bridge at ~0.5 GB while playing 6K, against 2.1 GB when cached frames kept them).
- No timeline semaphores: the bridge synchronises its stream before announcing a frame, and the
  client gives a buffer back only once its GPU copy has completed (the lease is dropped in
  `on_submitted_work_done`).
