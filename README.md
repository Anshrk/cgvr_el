# CGVR — Real-Time GPU SPH Dam-Break Simulation

A real-time, GPU-accelerated Smoothed Particle Hydrodynamics (SPH) fluid simulation written in
Rust, built for a university **CGVR** (Computer Graphics & VR) / **PAGP** (Parallel Architecture
& GPU Programming) combined coursework project.

The scenario is a dam-break / urban-flood simulation: a column of water particles collapses under
gravity, driven by a GPU compute pipeline, and rendered in real time.

## Context

The original project pitch (see [`context.md`](./context.md)) proposed a CUDA + DirectX12/Unity +
OpenXR pipeline targeting a 90 FPS VR headset demo with 60,000+ particles. That was adapted for
what's actually buildable and demoable on the development hardware (a MacBook with an Apple M2 —
no CUDA, no VR headset, no SteamVR/Quest-Link runtime on macOS) while keeping the project
legitimately satisfying both course components:

- **PAGP** → GPU compute shaders written in WGSL via [`wgpu`](https://wgpu.rs/), dispatched as
  genuinely parallel GPU kernels (the WGSL/wgpu analogue of CUDA kernels / HLSL compute shaders),
  running on Metal.
- **CGVR** → real-time 3D rendering, with a screen-space fluid rendering pipeline (in progress —
  see Status below). Stereoscopic/VR rendering is deprioritized to the end of the project (desktop
  stereo simulation, not a real OpenXR headset integration — no headset is available) in favor of
  spending the time on simulation and rendering quality, which is the more substantial deliverable.

The full implementation plan — architecture, milestone sequence, and what's explicitly out of
scope — lives in this repo's development history / project plan and is summarized below.

## Status

| Milestone | What it is | Status |
|---|---|---|
| M0 | Window + wgpu/Metal surface setup | ✅ Done |
| M1 | Naive O(n²) SPH solver (density/pressure + force/integration), dam-break init, flat-shaded point-sprite debug rendering | ✅ Done |
| M2 | GPU uniform spatial hash grid + bitonic sort neighbor search (replaces the O(n²) walk at scale) | ✅ Done |
| M3 | Screen-space fluid rendering (depth splat → bilateral blur → reconstructed normals → shading) + background/ground plane | ✅ Done — confirmed looking better than the debug view |
| M5 | Barricade placement (mouse-ray) + collision | ✅ Done |
| M6 | Vorticity heatmap, Fresnel/refraction, dynamic CFL timestep, on-screen profiling | ⏳ Planned |
| M4 | Desktop stereo camera (VR-sim) | ⏳ Deprioritized to the end |

**Validated so far:** the SPH solver is physically stable (dam-break column collapses, hits the
floor, splashes, settles — no NaN/blowup over 60+ seconds of simulated time), and the grid-based
neighbor search (M2) produces matching physics to the naive baseline (M1) while being
measurably faster — benchmarked at **~1.5× at 19k particles, ~10.5× at 53k particles** (GPU-synced
timing; see `src/app.rs` for the measurement methodology). Press **G** at runtime to toggle
between the two neighbor-search strategies and see this difference live.

The solver advances 8 fixed-`dt` substeps per rendered frame (`SUBSTEPS_PER_FRAME` in
`sim/mod.rs`) so simulated time keeps pace with wall-clock time at a small, numerically stable
timestep, rather than crawling at `dt`-per-frame speed.

Rendering defaults to the M1 debug view (flat-shaded billboard spheres); press **F** to switch to
the M3 screen-space fluid rendering pipeline (depth splat → bilateral blur → reconstructed normals
→ shading), confirmed to visually read as water rather than loose particles.

Barricades (M5) can be placed at runtime with a right-click: a ray is cast from the camera through
the cursor, intersected with the domain floor, and a fixed-size box obstacle is placed there — the
fluid collides with it (and with up to 7 other placed barricades) in the same compute pass as the
domain-wall collision.

## Running it

Requires a reasonably recent Rust toolchain (this project needs `rustc` ≥ 1.87 for the `wgpu`
version in use):

```sh
rustup update stable
cargo run
```

For a real performance test (not a debug build), use:

```sh
cargo run --release
```

### Controls

| Input | Action |
|---|---|
| Left-click + drag | Orbit the camera |
| Scroll wheel | Zoom in/out |
| Right-click | Place a barricade (ray-cast onto the floor at the cursor) — up to 8 at once, oldest replaced |
| `G` | Toggle neighbor-search strategy (grid-accelerated vs. naive O(n²)) — watch the FPS change |
| `F` | Toggle rendering mode: M1 point-sprite debug view vs. M3 screen-space fluid rendering (SSFR) |
| `R` | Reset the simulation to its initial dam-break state and clear all barricades |
| `P` | Pause/resume the simulation (camera controls keep working while paused) |
| `C` | Reset the camera to its default orbit view |
| `+` / `-` | Speed up / slow down the simulation's time scale (substeps per rendered frame) |
| `Q` | Quit |

The app logs particle count, frame time, and basic physics sanity stats (min/max height, max
speed, max density, NaN count) to stderr roughly once a second, plus densely for the first ~30
frames at startup (useful for spotting instability early).

## Project layout

```
src/
  main.rs          — winit event loop entry point
  app.rs           — window/input/frame-orchestration glue
  gpu.rs           — wgpu instance/adapter/device/surface setup
  camera.rs        — orbit camera + billboard camera uniform
  sim/             — SPH solver: particle buffers, compute pipelines (naive + grid), dam-break init
  render/
    background.rs  — ground plane + boundary wall geometry (opaque pass)
    point_sprite.rs— M1 debug view: flat-shaded billboard spheres
    ssfr/          — M3 screen-space fluid rendering: depth_splat, bilateral_blur, normals, shading
  shaders/         — all WGSL compute + render shaders
```

Each neighbor-search strategy (naive vs. grid) has its own WGSL shader pair
(`density_pressure.wgsl` / `forces_integrate.wgsl` vs. `density_pressure_grid.wgsl` /
`forces_integrate_grid.wgsl`) implementing identical SPH physics — they only differ in how
neighbors are found, which is what makes the M1-vs-M2 performance comparison meaningful.

The `Particle` struct (`sim/particle.rs`) has a compile-time assertion that it's exactly 48
bytes — WGSL's implicit struct padding rules (every `vec3<f32>` aligns to 16 bytes, even as a
struct's last field) don't always match Rust's natural layout, and a silent mismatch there
corrupts every particle read on the GPU side. If you change that struct's fields, update the
matching WGSL `struct Particle` in every shader that uses it and re-check the byte layout by hand.

## Why Rust + wgpu instead of CUDA/C++

Rust+CUDA tooling is immature and painful; `wgpu`'s WGSL compute shaders are the idiomatic way to
get real parallel GPU compute in Rust, and they run natively on Metal (this machine's GPU backend)
as well as Vulkan/DX12 elsewhere — without requiring an NVIDIA GPU. The compute-dispatch structure
(hash → sort → cell-table → neighbor pass) mirrors the classic CUDA reference architecture
(Green, 2010, *Particle Simulation using CUDA*), just expressed in WGSL.

## References

- Müller, Charypar, Gross (2003) — *Particle-based fluid simulation for interactive applications*
- Green (2010) — *Particle Simulation using CUDA* (NVIDIA whitepaper) — the hash/sort/cell-table
  neighbor-search architecture this project's M2 is based on
- van der Laan, Green, Sainz (2009) — *Screen space fluid rendering with curvature flow* — the
  basis for the M3 rendering pipeline
