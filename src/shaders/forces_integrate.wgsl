// M1: naive O(n^2) neighbor search (see density_pressure.wgsl). Reads the
// "current" particle buffer (unchanged since the density pass only touched
// density/pressure, not position/velocity) and writes the fully updated
// particle into the "next" ping-pong buffer, which becomes current next frame.

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
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;

const PI: f32 = 3.14159265358979;
const EPSILON: f32 = 1.0e-6;

// Magnitude of the Spiky kernel gradient (Desbrun & Cani), direction applied
// by the caller; the leading minus sign of the true gradient is folded in here.
fn spiky_grad_coeff(r: f32, h: f32) -> f32 {
    if (r > h || r < EPSILON) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    let term = h - r;
    return -45.0 / (PI * h6) * term * term;
}

// Viscosity kernel Laplacian (Muller et al. 2003).
fn visc_laplacian(r: f32, h: f32) -> f32 {
    if (r > h) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    return 45.0 / (PI * h6) * (h - r);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }

    let h = params.smoothing_radius;
    let p_i = particles_src[i];

    var pressure_force = vec3<f32>(0.0, 0.0, 0.0);
    var viscosity_force = vec3<f32>(0.0, 0.0, 0.0);
    var vorticity = 0.0;

    let n = params.particle_count;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let p_j = particles_src[j];
        let diff = p_i.position - p_j.position;
        let r = length(diff);
        if (r < EPSILON || r > h) {
            continue;
        }
        let dir = diff / r;
        let density_j = max(p_j.density, EPSILON);

        let grad_coeff = spiky_grad_coeff(r, h);
        let pressure_term = (p_i.pressure + p_j.pressure) / (2.0 * density_j);
        pressure_force = pressure_force - params.particle_mass * pressure_term * grad_coeff * dir;

        let vel_diff = p_j.velocity - p_i.velocity;
        viscosity_force = viscosity_force
            + params.viscosity * params.particle_mass * (vel_diff / density_j) * visc_laplacian(r, h);

        // Curl estimate (sum of (m_j/rho_j)(v_j - v_i) x gradW) drives the
        // vorticity heatmap visualization added in M6.
        let grad_w = grad_coeff * dir;
        let curl_term = cross(vel_diff, grad_w) * (params.particle_mass / density_j);
        vorticity = vorticity + length(curl_term);
    }

    let density_safe = max(p_i.density, EPSILON);
    var acceleration = (pressure_force + viscosity_force) / density_safe;
    acceleration.y = acceleration.y + params.gravity_y;

    var velocity = p_i.velocity + acceleration * params.dt;

    // Safety clamp against single-step numerical blowup (e.g. a transient
    // near-degenerate neighbor configuration). A properly tuned stiffness/dt
    // pair should rarely hit this; it is a robustness net, not the primary
    // stability mechanism. Revisit alongside the dynamic CFL timestep (M6).
    let max_speed = 15.0;
    let speed = length(velocity);
    if (speed > max_speed) {
        velocity = velocity * (max_speed / speed);
    }

    var position = p_i.position + velocity * params.dt;

    // Domain boundary collision: clamp position and reflect the offending
    // velocity component with damping. Placed barricades (M5) reuse this
    // same per-axis AABB test against their own bounds.
    let damping = 0.5;
    let margin = 0.001;
    if (position.x < params.domain_min.x) {
        position.x = params.domain_min.x + margin;
        velocity.x = -velocity.x * damping;
    }
    if (position.x > params.domain_max.x) {
        position.x = params.domain_max.x - margin;
        velocity.x = -velocity.x * damping;
    }
    if (position.y < params.domain_min.y) {
        position.y = params.domain_min.y + margin;
        velocity.y = -velocity.y * damping;
    }
    if (position.y > params.domain_max.y) {
        position.y = params.domain_max.y - margin;
        velocity.y = -velocity.y * damping;
    }
    if (position.z < params.domain_min.z) {
        position.z = params.domain_min.z + margin;
        velocity.z = -velocity.z * damping;
    }
    if (position.z > params.domain_max.z) {
        position.z = params.domain_max.z - margin;
        velocity.z = -velocity.z * damping;
    }

    var out: Particle;
    out.position = position;
    out.velocity = velocity;
    out.density = p_i.density;
    out.pressure = p_i.pressure;
    out.vorticity = vorticity;
    particles_dst[i] = out;
}
