pub const MAX_DEBRIS: usize = 16;
/// Scale factor for encoding the GPU force accumulator as fixed-point i32
/// (WGSL has no `atomic<f32>`). Chosen with wide headroom against i32
/// overflow even under a pessimistic particle pile-up (including against a
/// house's larger contact area) — see the design plan for the arithmetic.
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

/// What a debris object looks/behaves like. All three share the exact same
/// GPU collision + two-way force-coupling code — they only differ in shape,
/// mass (which controls how much the accumulated fluid impulse actually
/// moves them: a house's mass is high enough that it's effectively static,
/// while a tree is light enough to topple/get swept), and render color.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DebrisKind {
    Crate,
    Tree,
    House,
}

impl DebrisKind {
    pub fn half_extents(self) -> [f32; 3] {
        match self {
            DebrisKind::Crate => [0.2, 0.2, 0.2],
            DebrisKind::Tree => [0.12, 0.55, 0.12], // tall and thin
            DebrisKind::House => [0.45, 0.35, 0.45], // wide and squat
        }
    }

    pub fn mass(self) -> f32 {
        match self {
            // Heavier than a naive guess: with many SPH particles able to
            // overlap a small box across a whole frame's worth of substeps
            // before its position updates, the accumulated impulse can
            // spike large relative to mass, causing jarring per-frame
            // velocity jumps. More inertia + the speed clamp/damping in
            // `update_debris_from_gpu` are what keeps motion smooth.
            DebrisKind::Crate => 8.0,
            DebrisKind::Tree => 6.0, // lighter — knocked over/swept more easily
            // Heavy enough that impulse/mass is negligible: houses
            // participate in collision (water flows around/off them) but
            // don't meaningfully relocate, same as a real flooded house.
            // (400 wasn't enough — houses' larger footprint means more
            // contact area/particles, so the raw accumulated impulse is
            // itself bigger too; empirically verified 400 still let them
            // drift ~1m, same order as crates.)
            DebrisKind::House => 5000.0,
        }
    }

    pub fn color(self) -> [f32; 3] {
        match self {
            DebrisKind::Crate => [0.55, 0.45, 0.3],
            DebrisKind::Tree => [0.18, 0.45, 0.2],
            DebrisKind::House => [0.62, 0.32, 0.26],
        }
    }
}

/// How much damage a structure has taken. Houses barely move (by design —
/// see `DebrisKind::mass`), so displacement alone can't measure destruction
/// for them; `peak_force` (the hardest single-frame hit the object has ever
/// taken, independent of whether it was heavy enough to actually move) is
/// what makes a damage classification meaningful for anchored structures,
/// not just swept ones.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DestructionStatus {
    Intact,
    Damaged,
    Destroyed,
}

impl DebrisKind {
    /// (damaged_at, destroyed_at) peak-force thresholds (newtons-ish — this
    /// sim's force units, impulse/dt, not calibrated against real material
    /// strengths). Tuned empirically by logging observed peak forces during
    /// a real dam-break run: a tree at the front of the flood path (closest
    /// to the dam) peaked around 2.5k N on first contact and crossed 6k N
    /// within a couple seconds of sustained battering; crates (more contact
    /// area than a tree, less than a house) needed closer to 9k N; houses —
    /// despite being effectively immobile (`mass = 5000`) — still see
    /// forces in the same order of magnitude as crates (force here comes
    /// from contact area/particle pile-up, not mass), so they need a
    /// noticeably higher threshold to actually register as "destroyed"
    /// rather than just "damaged" within a typical demo run — reflecting
    /// that they're built sturdier, not that they're physically harder to
    /// hit.
    pub fn force_thresholds(self) -> (f32, f32) {
        match self {
            DebrisKind::Tree => (1200.0, 6000.0),
            DebrisKind::Crate => (2500.0, 9000.0),
            DebrisKind::House => (8000.0, 20000.0),
        }
    }
}

/// Live-editable copy of each kind's (damaged_at, destroyed_at) thresholds,
/// seeded from `DebrisKind::force_thresholds`'s defaults. Kept outside
/// `DebrisKind` (which stays a pure, copy-able descriptor) so the settings
/// panel can retune destruction sensitivity at runtime — e.g. during a demo
/// Q&A — without recompiling.
#[derive(Clone, Copy)]
pub struct ForceThresholds {
    pub tree: (f32, f32),
    pub crate_: (f32, f32),
    pub house: (f32, f32),
}

impl Default for ForceThresholds {
    fn default() -> Self {
        Self {
            tree: DebrisKind::Tree.force_thresholds(),
            crate_: DebrisKind::Crate.force_thresholds(),
            house: DebrisKind::House.force_thresholds(),
        }
    }
}

impl ForceThresholds {
    pub fn get(&self, kind: DebrisKind) -> (f32, f32) {
        match kind {
            DebrisKind::Tree => self.tree,
            DebrisKind::Crate => self.crate_,
            DebrisKind::House => self.house,
        }
    }
}

/// CPU-side debris state: the GPU-visible `Debris` plus bookkeeping that
/// never needs to be uploaded (spawn position/kind for the flood-impact
/// metric, color for rendering — the GPU side only needs collision/physics
/// data).
#[derive(Clone, Copy)]
pub struct DebrisState {
    pub gpu: Debris,
    pub spawn_position: [f32; 3],
    pub color: [f32; 3],
    pub kind: DebrisKind,
    /// Hardest single-frame force this object has experienced so far this
    /// run (updated in `SphSim::update_debris_from_gpu`), independent of
    /// how far it's actually moved — the basis for `destruction_status`.
    pub peak_force: f32,
}

impl DebrisState {
    pub fn new(center: [f32; 3], kind: DebrisKind) -> Self {
        Self {
            gpu: Debris {
                center,
                _pad0: 0.0,
                half_extents: kind.half_extents(),
                _pad1: 0.0,
                velocity: [0.0, 0.0, 0.0],
                mass: kind.mass(),
            },
            spawn_position: center,
            color: kind.color(),
            kind,
            peak_force: 0.0,
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

    /// Straight-line distance from the dam wall (the domain's `-X` face,
    /// where the collapsing water column starts) to this object's *spawn*
    /// position — the "how far from the source" axis for the
    /// destruction-vs-distance measurement. Fixed at spawn time (not
    /// current position) so the measurement reflects where the structure
    /// actually stood, not wherever the flood has since dragged it.
    pub fn distance_from_dam(&self, domain_min: [f32; 3]) -> f32 {
        self.spawn_position[0] - domain_min[0]
    }

    pub fn destruction_status(&self, thresholds: &ForceThresholds) -> DestructionStatus {
        let (damaged_at, destroyed_at) = thresholds.get(self.kind);
        if self.peak_force >= destroyed_at || self.displacement() > SWEPT_THRESHOLD * 3.0 {
            DestructionStatus::Destroyed
        } else if self.peak_force >= damaged_at || self.displacement() > SWEPT_THRESHOLD {
            DestructionStatus::Damaged
        } else {
            DestructionStatus::Intact
        }
    }

    /// The color to actually render this object as — `color` tinted toward
    /// char/soot as it takes damage, so destruction is visible in the scene
    /// itself and not just the settings-panel table.
    pub fn display_color(&self, thresholds: &ForceThresholds) -> [f32; 3] {
        const DAMAGED_TINT: [f32; 3] = [0.12, 0.08, 0.06];
        const DESTROYED_TINT: [f32; 3] = [0.05, 0.04, 0.04];
        let mix = |a: [f32; 3], b: [f32; 3], t: f32| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };
        match self.destruction_status(thresholds) {
            DestructionStatus::Intact => self.color,
            DestructionStatus::Damaged => mix(self.color, DAMAGED_TINT, 0.5),
            DestructionStatus::Destroyed => mix(self.color, DESTROYED_TINT, 0.85),
        }
    }
}

/// Fixed scripted spawn layout for v1 (pre-placed, not user-triggered — see
/// the project plan for the later mouse-placement extension path, mirroring
/// `place_barricade_at_cursor`). Positions sit in the dam-break's flow path:
/// `domain_min`/`domain_max` come from `SimParams`, matching the existing
/// dam-break block's rough footprint in `particle::init_dam_break`. A small
/// "street" of houses sits further downstream, with trees and crates
/// scattered in front of and among them — a miniature urban-flood scene.
pub fn spawn_layout(domain_min: [f32; 3], domain_max: [f32; 3]) -> Vec<DebrisState> {
    let floor_y = domain_min[1];
    let x_span = domain_max[0] - domain_min[0];
    let z_span = domain_max[2] - domain_min[2];

    let place = |spawns: &mut Vec<DebrisState>, kind: DebrisKind, xf: f32, zf: f32| {
        if spawns.len() >= MAX_DEBRIS {
            return;
        }
        let x = domain_min[0] + x_span * xf;
        let z = domain_min[2] + z_span * zf;
        let half_y = kind.half_extents()[1];
        spawns.push(DebrisState::new([x, floor_y + half_y, z], kind));
    };

    let mut spawns = Vec::with_capacity(MAX_DEBRIS);

    // Trees scattered in the flood's immediate path.
    for &(xf, zf) in &[(0.35, 0.3), (0.4, 0.7), (0.55, 0.45), (0.6, 0.8)] {
        place(&mut spawns, DebrisKind::Tree, xf, zf);
    }
    // Crates mixed in among them.
    for &(xf, zf) in &[(0.45, 0.2), (0.5, 0.6), (0.65, 0.35)] {
        place(&mut spawns, DebrisKind::Crate, xf, zf);
    }
    // A small row of houses further downstream — heavy enough to mostly
    // hold their ground while the water flows around/through them.
    for &(xf, zf) in &[(0.78, 0.25), (0.78, 0.5), (0.78, 0.75), (0.9, 0.5)] {
        place(&mut spawns, DebrisKind::House, xf, zf);
    }

    spawns
}
