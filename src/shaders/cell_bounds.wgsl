// M2 step 3/4: build the per-cell [start, end) range table from the now
// cell-key-sorted particle_indices array. No atomics needed: since the
// array is sorted, a cell's particles are exactly one contiguous run, so a
// single pass need only detect where each run begins/ends (Green, 2010).

struct GridParams {
    cell_size: f32,
    particle_count: u32,
    padded_count: u32,
    total_cells: u32,
    domain_min: vec3<f32>,
    grid_dim: vec3<u32>,
};

@group(0) @binding(0) var<uniform> grid: GridParams;
@group(0) @binding(1) var<storage, read> cell_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_end: array<u32>;

const SENTINEL: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(64)
fn clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= grid.total_cells) {
        return;
    }
    cell_start[cell] = SENTINEL;
    cell_end[cell] = SENTINEL;
}

@compute @workgroup_size(64)
fn bounds_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= grid.padded_count) {
        return;
    }

    let key = cell_keys[idx];
    // Padding slots carry a sentinel key far outside [0, total_cells) and
    // are never a real cell's neighbor-search target.
    if (key >= grid.total_cells) {
        return;
    }

    if (idx == 0u || cell_keys[idx - 1u] != key) {
        cell_start[key] = idx;
    }
    if (idx + 1u >= grid.padded_count || cell_keys[idx + 1u] != key) {
        cell_end[key] = idx + 1u;
    }
}
