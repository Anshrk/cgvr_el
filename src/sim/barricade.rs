pub const MAX_BARRICADES: usize = 8;
/// Fixed obstacle size for all placed barricades — simple enough to not need
/// a separate "scale" UI control for a first pass.
pub const BARRICADE_HALF_EXTENTS: [f32; 3] = [0.3, 0.5, 0.3];

/// Mirrors `struct Barricade` in `forces_integrate.wgsl` / `forces_integrate_grid.wgsl`.
/// Axis-aligned box: both the collision test and the render geometry (see
/// `render/background.rs`) treat it as a plain AABB.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Barricade {
    pub center: [f32; 3],
    pub _pad0: f32,
    pub half_extents: [f32; 3],
    pub _pad1: f32,
}

const _: () = assert!(std::mem::size_of::<Barricade>() == 32);

/// Mirrors `struct BarricadeListParams` in the same shaders — just the
/// active count, since the storage buffer itself is fixed-capacity
/// (`MAX_BARRICADES`) and unused slots are simply never read.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BarricadeListParams {
    pub count: u32,
    pub _pad: [u32; 3],
}
