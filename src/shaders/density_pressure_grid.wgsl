// M2 density/pressure pass: identical physics to density_pressure.wgsl (the
// M1 naive version, kept for the profiled neighbor-search comparison), but
// walks only the 27 grid cells that could contain a particle within one
// smoothing radius, via the hash/sort/cell-table built this frame, instead
// of every other particle.

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

struct GridParams {
    cell_size: f32,
    particle_count: u32,
    padded_count: u32,
    total_cells: u32,
    domain_min: vec3<f32>,
    grid_dim: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(1) @binding(0) var<uniform> grid: GridParams;
@group(1) @binding(1) var<storage, read> particle_indices: array<u32>;
@group(1) @binding(2) var<storage, read> cell_start: array<u32>;
@group(1) @binding(3) var<storage, read> cell_end: array<u32>;

const PI: f32 = 3.14159265358979;
const SENTINEL: u32 = 0xFFFFFFFFu;

fn poly6(r2: f32, h: f32) -> f32 {
    let h2 = h * h;
    if (r2 > h2) {
        return 0.0;
    }
    let term = h2 - r2;
    let h9 = h2 * h2 * h2 * h2 * h;
    return (315.0 / (64.0 * PI * h9)) * term * term * term;
}

fn cell_coord(position: vec3<f32>) -> vec3<i32> {
    let rel = (position - grid.domain_min) / grid.cell_size;
    return vec3<i32>(floor(rel));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.particle_count) {
        return;
    }

    let h = params.smoothing_radius;
    let position_i = particles[i].position;
    let base = cell_coord(position_i);
    let dim = vec3<i32>(grid.grid_dim);

    var density: f32 = 0.0;

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
                    let diff = position_i - particles[j].position;
                    let r2 = dot(diff, diff);
                    density = density + params.particle_mass * poly6(r2, h);
                }
            }
        }
    }

    particles[i].density = density;
    particles[i].pressure = max(params.stiffness * (density - params.rest_density), 0.0);
}
