/// Uniform spatial hash grid sizing + the static bitonic-sort dispatch
/// schedule. Both are fully determined by the (fixed, for a given run)
/// particle count and domain/cell size, so they're computed once at startup
/// rather than per frame.
#[derive(Clone, Copy)]
pub struct GridInfo {
    pub cell_size: f32,
    pub dims: [u32; 3],
    pub total_cells: u32,
    pub padded_particle_count: u32,
}

impl GridInfo {
    pub fn new(domain_min: [f32; 3], domain_max: [f32; 3], cell_size: f32, particle_count: u32) -> Self {
        let dims = [0, 1, 2].map(|axis| {
            let extent = domain_max[axis] - domain_min[axis];
            ((extent / cell_size).ceil() as u32).max(1)
        });
        let total_cells = dims[0] * dims[1] * dims[2];
        let padded_particle_count = particle_count.max(1).next_power_of_two();

        Self {
            cell_size,
            dims,
            total_cells,
            padded_particle_count,
        }
    }
}

/// Mirrors `GridParams` in `shaders/hash_assign.wgsl` / `cell_bounds.wgsl` /
/// `density_pressure_grid.wgsl` / `forces_integrate_grid.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParams {
    pub cell_size: f32,
    pub particle_count: u32,
    pub padded_count: u32,
    pub total_cells: u32,
    pub domain_min: [f32; 3],
    pub _pad0: f32,
    pub grid_dim: [u32; 3],
    pub _pad1: u32,
}

/// The (k, j) compare-exchange stage schedule for an iterative bitonic sort
/// over `n` elements (`n` must be a power of two). Static for a fixed `n`,
/// so computed once and reused as the per-stage dispatch list every frame.
pub fn bitonic_stage_pairs(n: u32) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    let mut k = 2;
    while k <= n {
        let mut j = k / 2;
        while j >= 1 {
            pairs.push((k, j));
            j /= 2;
        }
        k *= 2;
    }
    pairs
}

/// Mirrors `BitonicParams` in `shaders/bitonic_sort.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BitonicParams {
    pub k: u32,
    pub j: u32,
    pub n: u32,
    pub _pad: u32,
}
