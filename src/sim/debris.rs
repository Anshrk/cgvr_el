pub const MAX_DEBRIS: usize = 8;
/// Smaller than barricades — this is loose clutter the flood carries, not a wall.
pub const DEBRIS_HALF_EXTENTS: [f32; 3] = [0.2, 0.2, 0.2];
/// Scale factor for encoding the GPU force accumulator as fixed-point i32
/// (WGSL has no `atomic<f32>`). Chosen with ~50x headroom against i32
/// overflow even under a pessimistic particle pile-up — see debris.rs's
/// module docs in the design plan for the arithmetic.
pub const FORCE_FIXED_POINT_SCALE: f32 = 1000.0;
/// Displacement (meters) from spawn position counted as "swept away" for
/// the flood-impact metric.
pub const SWEPT_THRESHOLD: f32 = 0.5;

/// Mirrors `struct Debris` in `forces_integrate.wgsl` / `forces_integrate_grid.wgsl`.
/// Same (vec3, trailing-scalar) packing trick as `Particle`/`Barricade`: each
/// pair fills one 16-byte slot, so `mass` costs nothing extra after `velocity`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Debris {
    pub center: [f32; 3],
    pub _pad0: f32,
    pub half_extents: [f32; 3],
    pub _pad1: f32,
    pub velocity: [f32; 3],
    pub mass: f32,
}

const _: () = assert!(std::mem::size_of::<Debris>() == 48);

/// Mirrors `struct DebrisListParams` — same shape as `BarricadeListParams`
/// (separate scalar padding fields, not `array<u32,3>`: WGSL requires array
/// strides inside a uniform-address-space struct to be multiples of 16).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebrisListParams {
    pub count: u32,
    pub _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<DebrisListParams>() == 16);

/// CPU-side debris state: the GPU-visible `Debris` plus bookkeeping that
/// never needs to be uploaded (spawn position, for the flood-impact metric).
#[derive(Clone, Copy)]
pub struct DebrisState {
    pub gpu: Debris,
    pub spawn_position: [f32; 3],
}

impl DebrisState {
    pub fn new(center: [f32; 3], mass: f32) -> Self {
        Self {
            gpu: Debris {
                center,
                _pad0: 0.0,
                half_extents: DEBRIS_HALF_EXTENTS,
                _pad1: 0.0,
                velocity: [0.0, 0.0, 0.0],
                mass,
            },
            spawn_position: center,
        }
    }

    pub fn displacement(&self) -> f32 {
        let d = [
            self.gpu.center[0] - self.spawn_position[0],
            self.gpu.center[1] - self.spawn_position[1],
            self.gpu.center[2] - self.spawn_position[2],
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
}

/// Fixed scripted spawn layout for v1 (pre-placed, not user-triggered — see
/// the project plan for the later mouse-placement extension path, mirroring
/// `place_barricade_at_cursor`). Positions sit in the dam-break's flow path:
/// `domain_min`/`domain_max` come from `SimParams`, matching the existing
/// dam-break block's rough footprint in `particle::init_dam_break`.
pub fn spawn_layout(domain_min: [f32; 3], domain_max: [f32; 3]) -> Vec<DebrisState> {
    let floor_y = domain_min[1];
    let half_y = DEBRIS_HALF_EXTENTS[1];
    // Heavier than a naive "crate" guess: with many SPH particles able to
    // overlap a small (0.2 half-extent) box across a whole frame's worth of
    // substeps before its position updates, the accumulated impulse can
    // spike large relative to mass, causing jarring per-frame velocity
    // jumps rather than a smooth push. More inertia + the speed clamp/
    // damping in `update_debris_from_gpu` are the fix.
    let mass = 8.0_f32;

    let x_span = domain_max[0] - domain_min[0];
    let z_span = domain_max[2] - domain_min[2];

    // Scatter a handful of crates downstream of the dam, roughly where the
    // collapsing column will reach them.
    let xs = [0.35_f32, 0.5, 0.65, 0.8];
    let zs = [0.35_f32, 0.65];

    let mut spawns = Vec::with_capacity(MAX_DEBRIS);
    'outer: for &zf in &zs {
        for &xf in &xs {
            if spawns.len() >= MAX_DEBRIS {
                break 'outer;
            }
            let x = domain_min[0] + x_span * xf;
            let z = domain_min[2] + z_span * zf;
            spawns.push(DebrisState::new([x, floor_y + half_y, z], mass));
        }
    }
    spawns
}
