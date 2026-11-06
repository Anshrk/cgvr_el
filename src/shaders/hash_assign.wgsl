// M2 step 1/4: assign each particle to a uniform grid cell (cell size = h,
// so the 27-neighbor-cell walk in density_pressure_grid.wgsl /
// forces_integrate_grid.wgsl is guaranteed to cover every particle within
// one smoothing radius). Padding slots (index >= particle_count, needed to
// round up to a power of two for the bitonic sort) get a sentinel key so
// they always sort to the end and are never treated as real neighbors.

struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
    vorticity: f32,
};

struct GridParams {
    cell_size: f32,
    particle_count: u32,
    padded_count: u32,
    total_cells: u32,
    domain_min: vec3<f32>,
    grid_dim: vec3<u32>,
};

@group(0) @binding(0) var<uniform> grid: GridParams;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> cell_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> particle_indices: array<u32>;

const SENTINEL: u32 = 0xFFFFFFFFu;

fn clamp_cell_coord(position: vec3<f32>) -> vec3<u32> {
    let rel = (position - grid.domain_min) / grid.cell_size;
    let coord = vec3<i32>(floor(rel));
    let dim = vec3<i32>(grid.grid_dim);
    let clamped = clamp(coord, vec3<i32>(0, 0, 0), dim - vec3<i32>(1, 1, 1));
    return vec3<u32>(clamped);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= grid.padded_count) {
        return;
    }

    particle_indices[i] = i;

    if (i >= grid.particle_count) {
        cell_keys[i] = SENTINEL;
        return;
    }

    let coord = clamp_cell_coord(particles[i].position);
    cell_keys[i] = coord.x + coord.y * grid.grid_dim.x + coord.z * grid.grid_dim.x * grid.grid_dim.y;
}
