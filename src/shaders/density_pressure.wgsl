// M1: naive O(n^2) neighbor search, correctness-first. Replaced by the GPU
// spatial hash grid neighbor walk in M2 without changing this kernel's math.

struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
    vorticity: f32,
};

struct SimParams {
    smoothing_radius: f32,
    rest_density: f32,
    stiffness: f32,
    viscosity: f32,
    particle_mass: f32,
    dt: f32,
    gravity_y: f32,
    particle_count: u32,
    domain_min: vec3<f32>,
    domain_max: vec3<f32>,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

const PI: f32 = 3.14159265358979;

// Poly6 smoothing kernel (Muller et al. 2003) for density estimation.
fn poly6(r2: f32, h: f32) -> f32 {
    let h2 = h * h;
    if (r2 > h2) {
        return 0.0;
    }
    let term = h2 - r2;
    let h9 = h2 * h2 * h2 * h2 * h;
    return (315.0 / (64.0 * PI * h9)) * term * term * term;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }

    let h = params.smoothing_radius;
    let position_i = particles[i].position;

    var density: f32 = 0.0;
    let n = params.particle_count;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let diff = position_i - particles[j].position;
        let r2 = dot(diff, diff);
        density = density + params.particle_mass * poly6(r2, h);
    }

    particles[i].density = density;
    particles[i].pressure = max(params.stiffness * (density - params.rest_density), 0.0);
}
