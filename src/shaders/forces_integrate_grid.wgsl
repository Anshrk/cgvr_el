// M2 force/integration pass: identical physics to forces_integrate.wgsl (the
// M1 naive version), but walks only the 27-neighbor-cell grid range instead
// of every other particle. See density_pressure_grid.wgsl for the shared
// grid-walk pattern and cell_bounds.wgsl for how the range table is built.

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
    cohesion: f32,
};

struct GridParams {
    cell_size: f32,
    particle_count: u32,
    padded_count: u32,
    total_cells: u32,
    domain_min: vec3<f32>,
    grid_dim: vec3<u32>,
};

// M5: user-placed obstacles (mouse-ray picked, see app.rs).
struct Barricade {
    center: vec3<f32>,
    half_extents: vec3<f32>,
};

struct BarricadeListParams {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;
@group(0) @binding(3) var<uniform> barricade_list: BarricadeListParams;
@group(0) @binding(4) var<storage, read> barricades: array<Barricade>;
@group(1) @binding(0) var<uniform> grid: GridParams;
@group(1) @binding(1) var<storage, read> particle_indices: array<u32>;
@group(1) @binding(2) var<storage, read> cell_start: array<u32>;
@group(1) @binding(3) var<storage, read> cell_end: array<u32>;

const PI: f32 = 3.14159265358979;
const EPSILON: f32 = 1.0e-6;
const SENTINEL: u32 = 0xFFFFFFFFu;

fn spiky_grad_coeff(r: f32, h: f32) -> f32 {
    if (r > h || r < EPSILON) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    let term = h - r;
    return -45.0 / (PI * h6) * term * term;
}

fn visc_laplacian(r: f32, h: f32) -> f32 {
    if (r > h) {
        return 0.0;
    }
    let h6 = h * h * h * h * h * h;
    return 45.0 / (PI * h6) * (h - r);
}

fn cell_coord(position: vec3<f32>) -> vec3<i32> {
    let rel = (position - grid.domain_min) / grid.cell_size;
    return vec3<i32>(floor(rel));
}

// M5: push a particle out of any barricade AABB it's penetrated, along
// whichever axis has the smallest penetration depth, reflecting that axis's
// velocity component (same style as the domain-wall collision below).
fn apply_barricade_collisions(position_in: vec3<f32>, velocity_in: vec3<f32>) -> array<vec3<f32>, 2> {
    var position = position_in;
    var velocity = velocity_in;
    let damping = 0.3;
    let margin = 0.001;

    for (var b: u32 = 0u; b < barricade_list.count; b = b + 1u) {
        let bar = barricades[b];
        let local = position - bar.center;
        if (abs(local.x) < bar.half_extents.x && abs(local.y) < bar.half_extents.y && abs(local.z) < bar.half_extents.z) {
            let pen_x = bar.half_extents.x - abs(local.x);
            let pen_y = bar.half_extents.y - abs(local.y);
            let pen_z = bar.half_extents.z - abs(local.z);

            if (pen_x <= pen_y && pen_x <= pen_z) {
                position.x = bar.center.x + sign(local.x) * (bar.half_extents.x + margin);
                velocity.x = -velocity.x * damping;
            } else if (pen_y <= pen_x && pen_y <= pen_z) {
                position.y = bar.center.y + sign(local.y) * (bar.half_extents.y + margin);
                velocity.y = -velocity.y * damping;
            } else {
                position.z = bar.center.z + sign(local.z) * (bar.half_extents.z + margin);
                velocity.z = -velocity.z * damping;
            }
        }
    }

    return array<vec3<f32>, 2>(position, velocity);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }

    let h = params.smoothing_radius;
    let p_i = particles_src[i];
    let base = cell_coord(p_i.position);
    let dim = vec3<i32>(grid.grid_dim);

    var pressure_force = vec3<f32>(0.0, 0.0, 0.0);
    var viscosity_force = vec3<f32>(0.0, 0.0, 0.0);
    var cohesion_force = vec3<f32>(0.0, 0.0, 0.0);
    var vorticity = 0.0;

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let coord = base + vec3<i32>(dx, dy, dz);
                if (any(coord < vec3<i32>(0, 0, 0)) || any(coord >= dim)) {
                    continue;
                }
                let cell = u32(coord.x) + u32(coord.y) * grid.grid_dim.x
                    + u32(coord.z) * grid.grid_dim.x * grid.grid_dim.y;
                let start = cell_start[cell];
                if (start == SENTINEL) {
                    continue;
                }
                let end = cell_end[cell];
                for (var k: u32 = start; k < end; k = k + 1u) {
                    let j = particle_indices[k];
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

                    // Surface-tension-like cohesion (see forces_integrate.wgsl's
                    // naive version for the rationale) — identical physics,
                    // just walked via the grid neighbor loop.
                    let cohesion_weight = (h - r) / h;
                    cohesion_force = cohesion_force - params.cohesion * params.particle_mass * cohesion_weight * dir;

                    let grad_w = grad_coeff * dir;
                    let curl_term = cross(vel_diff, grad_w) * (params.particle_mass / density_j);
                    vorticity = vorticity + length(curl_term);
                }
            }
        }
    }

    let density_safe = max(p_i.density, EPSILON);
    var acceleration = (pressure_force + viscosity_force + cohesion_force) / density_safe;
    acceleration.y = acceleration.y + params.gravity_y;

    var velocity = p_i.velocity + acceleration * params.dt;

    let max_speed = 15.0;
    let speed = length(velocity);
    if (speed > max_speed) {
        velocity = velocity * (max_speed / speed);
    }

    var position = p_i.position + velocity * params.dt;

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

    let barricade_result = apply_barricade_collisions(position, velocity);
    position = barricade_result[0];
    velocity = barricade_result[1];

    var out: Particle;
    out.position = position;
    out.velocity = velocity;
    out.density = p_i.density;
    out.pressure = p_i.pressure;
    out.vorticity = vorticity;
    particles_dst[i] = out;
}
