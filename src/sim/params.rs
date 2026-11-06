/// Simulation constants shared by every compute pass, uploaded once per frame.
///
/// Field order and padding mirror WGSL's layout rules exactly (vec3<f32> has
/// align 16 / size 12, so each vec3 here is paired with a trailing f32 that
/// WGSL packs into the same 16-byte slot): see `shaders/density_pressure.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub smoothing_radius: f32, // h
    pub rest_density: f32,
    pub stiffness: f32, // gas constant k in pressure = k * (density - rest_density)
    pub viscosity: f32, // dynamic viscosity coefficient mu
    pub particle_mass: f32,
    pub dt: f32,
    pub gravity_y: f32,
    pub particle_count: u32,
    pub domain_min: [f32; 3],
    pub _pad0: f32,
    pub domain_max: [f32; 3],
    // Short-range attraction between neighbors (surface-tension-like
    // cohesion) — without this, SPH water behaves like a spray of
    // independent particles instead of sticking into sheets/droplets.
    // Packs into the same 16-byte slot as domain_max (trailing-scalar-
    // after-vec3), so WGSL's struct size stays 64 bytes, not 80 — verify
    // with the layout arithmetic in forces_integrate.wgsl's comment before
    // adding more fields here.
    pub cohesion: f32,
}

// WGSL's SimParams struct (as declared in forces_integrate*.wgsl) strides
// at 64 bytes; a mismatch here silently corrupts every kernel reading this
// buffer.
const _: () = assert!(std::mem::size_of::<SimParams>() == 64);

impl SimParams {
    pub fn new_dam_break(particle_count: u32, spacing: f32) -> Self {
        let smoothing_radius = spacing * 1.3;
        let rest_density = 1000.0;
        let particle_mass = rest_density * spacing * spacing * spacing;

        Self {
            smoothing_radius,
            rest_density,
            stiffness: 40.0,
            viscosity: 1.5,
            particle_mass,
            dt: 0.001,
            gravity_y: -9.81,
            particle_count,
            domain_min: [-5.0, 0.0, -3.0],
            _pad0: 0.0,
            domain_max: [5.0, 6.0, 3.0],
            cohesion: 20.0,
        }
    }
}
