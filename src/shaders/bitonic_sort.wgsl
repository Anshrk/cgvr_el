// M2 step 2/4: one compare-exchange stage of an iterative bitonic sort over
// (cell_keys, particle_indices) pairs, keyed by cell_keys. The full sort is
// a fixed sequence of (k, j) stage dispatches (see sim::grid::bitonic_stage_pairs),
// each using its own tiny precomputed uniform buffer — no per-frame CPU
// writes needed since the schedule only depends on the (fixed) padded
// particle count.

struct BitonicParams {
    k: u32,
    j: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: BitonicParams;
@group(0) @binding(1) var<storage, read_write> cell_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> particle_indices: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }

    let partner = i ^ params.j;
    if (partner <= i || partner >= params.n) {
        return;
    }

    let ascending = (i & params.k) == 0u;
    let key_i = cell_keys[i];
    let key_p = cell_keys[partner];

    let should_swap = select((key_i < key_p), (key_i > key_p), ascending);
    if (should_swap) {
        cell_keys[i] = key_p;
        cell_keys[partner] = key_i;
        let idx_i = particle_indices[i];
        let idx_p = particle_indices[partner];
        particle_indices[i] = idx_p;
        particle_indices[partner] = idx_i;
    }
}
