use rand::RngExt;

use crate::sim::params::SimParams;

/// Mirrors `struct Particle` in `shaders/density_pressure.wgsl` /
/// `shaders/forces_integrate.wgsl` field-for-field, including WGSL's implicit
/// vec3-then-scalar packing (each (vec3, f32) pair occupies one 16-byte slot).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub density: f32,
    pub velocity: [f32; 3],
    pub pressure: f32,
    pub vorticity: f32,
    pub _pad: [f32; 3],
}

/// Fills a block near one side of the domain (a "dam") with a jittered
/// regular lattice of particles, standing in for the collapsing water column
/// in the dam-break scenario. Returns the generated particles; `params` is
/// updated in-place with the resulting exact particle count.
pub fn init_dam_break(params: &mut SimParams, spacing: f32) -> Vec<Particle> {
    let mut rng = rand::rng();

    let [min_x, min_y, min_z] = params.domain_min;
    let [_max_x, max_y, max_z] = params.domain_max;

    // Dam occupies a modest block near one wall: small enough to keep M1's
    // naive O(n^2) neighbor search fast to iterate on; M2's spatial hash grid
    // is what lets particle count scale up substantially.
    let dam_width = (params.domain_max[0] - params.domain_min[0]) * 0.22;
    let block_min = [min_x + spacing, min_y + spacing, min_z + spacing];
    let block_max = [
        min_x + dam_width,
        (max_y * 0.45).max(block_min[1] + spacing),
        max_z - spacing,
    ];

    let jitter = spacing * 0.05;
    let mut particles = Vec::new();

    let mut x = block_min[0];
    while x < block_max[0] {
        let mut y = block_min[1];
        while y < block_max[1] {
            let mut z = block_min[2];
            while z < block_max[2] {
                let jx = rng.random_range(-jitter..jitter);
                let jy = rng.random_range(-jitter..jitter);
                let jz = rng.random_range(-jitter..jitter);
                particles.push(Particle {
                    position: [x + jx, y + jy, z + jz],
                    density: 0.0,
                    velocity: [0.0, 0.0, 0.0],
                    pressure: 0.0,
                    vorticity: 0.0,
                    _pad: [0.0; 3],
                });
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }

    params.particle_count = particles.len() as u32;
    particles
}
