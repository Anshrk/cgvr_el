pub mod barricade;
pub mod grid;
pub mod params;
pub mod particle;

use wgpu::util::DeviceExt;

use barricade::{Barricade, BarricadeListParams, MAX_BARRICADES};
use grid::{bitonic_stage_pairs, BitonicParams, GridInfo, GridParams};
use params::SimParams;
use particle::{init_dam_break, Particle};

const WORKGROUP_SIZE: u32 = 64;
const DEFAULT_SUBSTEPS_PER_FRAME: u32 = 8;

// M6: dynamic CFL timestep, dt = clamp(CFL_NUMBER * h / max_speed, DT_MIN, DT_MAX).
// Updated on the existing periodic debug-stats readback cadence (~once/sec,
// or densely for the first ~30 frames — see App), not a per-frame GPU
// reduction: this sim's real stability limit is the SPH pressure stiffness
// term, not pure velocity advection, so a coarser update is enough to show
// dt actually shrink during a fast splash without adding a whole new GPU
// reduction + non-blocking-readback pipeline for a mostly-cosmetic gain.
// DT_MAX matches the fixed dt already validated stable at this stiffness.
const CFL_NUMBER: f32 = 0.02;
const DT_MIN: f32 = 0.0002;
const DT_MAX: f32 = 0.001;

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// Owns the SPH particle buffers (ping-ponged) and both neighbor-search
/// pipelines, and drives one simulation step per frame.
///
/// Two interchangeable neighbor-search strategies share the same density and
/// force/integration *physics* (each has a naive and a grid-accelerated
/// shader variant): M1's naive O(n^2) walk, kept around so particle-count
/// vs. frame-time can be profiled against M2's GPU uniform hash grid +
/// bitonic sort + cell table. `toggle_neighbor_search` swaps between them at
/// runtime for that comparison; the grid path is the default since it's what
/// lets particle count scale up.
pub struct SphSim {
    pub params: SimParams,
    params_buffer: wgpu::Buffer,
    use_grid: bool,
    spacing: f32,
    substeps_per_frame: u32,
    adaptive_dt: bool,

    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    current_is_a: bool,

    // --- M5: user-placed barricades ---
    barricades: Vec<Barricade>,
    barricade_buffer: wgpu::Buffer,
    barricade_count_buffer: wgpu::Buffer,

    // --- M1: naive O(n^2) neighbor search ---
    density_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    density_bind_group_a: wgpu::BindGroup,
    density_bind_group_b: wgpu::BindGroup,
    integrate_bind_group_a_to_b: wgpu::BindGroup,
    integrate_bind_group_b_to_a: wgpu::BindGroup,

    // --- M2: GPU uniform hash grid + bitonic sort ---
    grid_info: GridInfo,
    hash_assign_pipeline: wgpu::ComputePipeline,
    bitonic_pipeline: wgpu::ComputePipeline,
    cell_bounds_clear_pipeline: wgpu::ComputePipeline,
    cell_bounds_pipeline: wgpu::ComputePipeline,
    density_grid_pipeline: wgpu::ComputePipeline,
    integrate_grid_pipeline: wgpu::ComputePipeline,

    hash_assign_bind_group_a: wgpu::BindGroup,
    hash_assign_bind_group_b: wgpu::BindGroup,
    bitonic_bind_groups: Vec<wgpu::BindGroup>,
    cell_bounds_bind_group: wgpu::BindGroup,
    density_grid_bind_group_a: wgpu::BindGroup,
    density_grid_bind_group_b: wgpu::BindGroup,
    integrate_grid_bind_group_a_to_b: wgpu::BindGroup,
    integrate_grid_bind_group_b_to_a: wgpu::BindGroup,
    grid_data_bind_group: wgpu::BindGroup,

    // Kept alive: bitonic_bind_groups borrow these via as_entire_binding at
    // creation time, but the buffers themselves must outlive the bind groups.
    _bitonic_stage_buffers: Vec<wgpu::Buffer>,
}

impl SphSim {
    pub fn new(device: &wgpu::Device) -> Self {
        let spacing = 0.12_f32;
        let mut params = SimParams::new_dam_break(0, spacing);
        let particles = init_dam_break(&mut params, spacing);
        let particle_count = params.particle_count;

        let grid_info = GridInfo::new(
            params.domain_min,
            params.domain_max,
            params.smoothing_radius,
            particle_count,
        );

        // ---------- particle buffers (shared by both neighbor-search paths) ----------
        let buffer_size =
            (particles.len() * std::mem::size_of::<Particle>()) as wgpu::BufferAddress;
        let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles-a"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles-b"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ==================== M5: barricades (shared by both integrate paths) ====================
        let barricade_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("barricades"),
            size: (MAX_BARRICADES * std::mem::size_of::<Barricade>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let barricade_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("barricade-count"),
            contents: bytemuck::bytes_of(&BarricadeListParams { count: 0, _pad: [0; 3] }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ==================== M1: naive O(n^2) ====================
        let density_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density_pressure"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/density_pressure.wgsl").into(),
            ),
        });
        let integrate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces_integrate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/forces_integrate.wgsl").into(),
            ),
        });

        let density_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density-bgl"),
            entries: &[uniform_entry(0), storage_entry(1, false)],
        });
        let integrate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("integrate-bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
                storage_entry(4, true),
            ],
        });

        let density_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("density-pipeline-layout"),
                bind_group_layouts: &[Some(&density_bgl)],
                immediate_size: 0,
            });
        let integrate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrate-pipeline-layout"),
                bind_group_layouts: &[Some(&integrate_bgl)],
                immediate_size: 0,
            });

        let density_pipeline = make_pipeline(
            device,
            "density-pipeline",
            &density_pipeline_layout,
            &density_shader,
            "main",
        );
        let integrate_pipeline = make_pipeline(
            device,
            "integrate-pipeline",
            &integrate_pipeline_layout,
            &integrate_shader,
            "main",
        );

        let density_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density-bg-a"),
            layout: &density_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_a.as_entire_binding() },
            ],
        });
        let density_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density-bg-b"),
            layout: &density_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_b.as_entire_binding() },
            ],
        });

        let integrate_bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate-bg-a-to-b"),
            layout: &integrate_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buffer_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: barricade_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: barricade_buffer.as_entire_binding() },
            ],
        });
        let integrate_bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integrate-bg-b-to-a"),
            layout: &integrate_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buffer_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: barricade_count_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: barricade_buffer.as_entire_binding() },
            ],
        });

        // ==================== M2: GPU uniform hash grid + bitonic sort ====================
        let grid_params = GridParams {
            cell_size: grid_info.cell_size,
            particle_count,
            padded_count: grid_info.padded_particle_count,
            total_cells: grid_info.total_cells,
            domain_min: params.domain_min,
            _pad0: 0.0,
            grid_dim: grid_info.dims,
            _pad1: 0,
        };
        let grid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid-params"),
            contents: bytemuck::bytes_of(&grid_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let padded_count = grid_info.padded_particle_count;
        let cell_keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell-keys"),
            size: (padded_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let particle_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-indices"),
            size: (padded_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cell_start_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell-start"),
            size: (grid_info.total_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cell_end_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell-end"),
            size: (grid_info.total_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // --- hash_assign ---
        let hash_assign_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hash_assign"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hash_assign.wgsl").into()),
        });
        let hash_assign_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hash-assign-bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let hash_assign_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hash-assign-layout"),
            bind_group_layouts: &[Some(&hash_assign_bgl)],
            immediate_size: 0,
        });
        let hash_assign_pipeline = make_pipeline(
            device,
            "hash-assign-pipeline",
            &hash_assign_layout,
            &hash_assign_shader,
            "main",
        );
        let hash_assign_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash-assign-bg-a"),
            layout: &hash_assign_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_keys_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: particle_indices_buffer.as_entire_binding() },
            ],
        });
        let hash_assign_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hash-assign-bg-b"),
            layout: &hash_assign_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buffer_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_keys_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: particle_indices_buffer.as_entire_binding() },
            ],
        });

        // --- bitonic sort: one bind group per static (k, j) stage ---
        let bitonic_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bitonic_sort"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bitonic_sort.wgsl").into()),
        });
        let bitonic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bitonic-bgl"),
            entries: &[uniform_entry(0), storage_entry(1, false), storage_entry(2, false)],
        });
        let bitonic_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bitonic-layout"),
            bind_group_layouts: &[Some(&bitonic_bgl)],
            immediate_size: 0,
        });
        let bitonic_pipeline = make_pipeline(
            device,
            "bitonic-pipeline",
            &bitonic_layout,
            &bitonic_shader,
            "main",
        );

        let stage_pairs = bitonic_stage_pairs(padded_count);
        let mut bitonic_stage_buffers = Vec::with_capacity(stage_pairs.len());
        let mut bitonic_bind_groups = Vec::with_capacity(stage_pairs.len());
        for (k, j) in &stage_pairs {
            let stage_params = BitonicParams { k: *k, j: *j, n: padded_count, _pad: 0 };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bitonic-stage-params"),
                contents: bytemuck::bytes_of(&stage_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bitonic-stage-bg"),
                layout: &bitonic_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cell_keys_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: particle_indices_buffer.as_entire_binding() },
                ],
            });
            bitonic_stage_buffers.push(buf);
            bitonic_bind_groups.push(bg);
        }

        // --- cell_bounds (clear + bounds entry points share one bind group layout) ---
        let cell_bounds_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell_bounds"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cell_bounds.wgsl").into()),
        });
        let cell_bounds_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cell-bounds-bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ],
        });
        let cell_bounds_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cell-bounds-layout"),
            bind_group_layouts: &[Some(&cell_bounds_bgl)],
            immediate_size: 0,
        });
        let cell_bounds_clear_pipeline = make_pipeline(
            device,
            "cell-bounds-clear-pipeline",
            &cell_bounds_layout,
            &cell_bounds_shader,
            "clear_main",
        );
        let cell_bounds_pipeline = make_pipeline(
            device,
            "cell-bounds-pipeline",
            &cell_bounds_layout,
            &cell_bounds_shader,
            "bounds_main",
        );
        let cell_bounds_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cell-bounds-bg"),
            layout: &cell_bounds_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_keys_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_start_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_end_buffer.as_entire_binding() },
            ],
        });

        // --- grid-accelerated density/pressure + force/integrate ---
        let density_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density_pressure_grid"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/density_pressure_grid.wgsl").into(),
            ),
        });
        let integrate_grid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forces_integrate_grid"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/forces_integrate_grid.wgsl").into(),
            ),
        });

        // group1 (grid neighbor data) is shared, by the same BindGroupLayout
        // object, between the density_grid and integrate_grid pipelines.
        let grid_data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid-data-bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
            ],
        });
        let grid_data_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid-data-bg"),
            layout: &grid_data_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particle_indices_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_start_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_end_buffer.as_entire_binding() },
            ],
        });

        let density_grid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density-grid-layout"),
            bind_group_layouts: &[Some(&density_bgl), Some(&grid_data_bgl)],
            immediate_size: 0,
        });
        let integrate_grid_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("integrate-grid-layout"),
            bind_group_layouts: &[Some(&integrate_bgl), Some(&grid_data_bgl)],
            immediate_size: 0,
        });

        let density_grid_pipeline = make_pipeline(
            device,
            "density-grid-pipeline",
            &density_grid_layout,
            &density_grid_shader,
            "main",
        );
        let integrate_grid_pipeline = make_pipeline(
            device,
            "integrate-grid-pipeline",
            &integrate_grid_layout,
            &integrate_grid_shader,
            "main",
        );

        // density_grid / integrate_grid group0 reuses the exact same
        // per-buffer bind groups as the naive path (same bgl, same bindings).

        Self {
            params,
            params_buffer,
            use_grid: true,
            spacing,
            substeps_per_frame: DEFAULT_SUBSTEPS_PER_FRAME,
            adaptive_dt: true,

            buffer_a,
            buffer_b,
            current_is_a: true,

            barricades: Vec::new(),
            barricade_buffer,
            barricade_count_buffer,

            density_pipeline,
            integrate_pipeline,
            density_bind_group_a: density_bind_group_a.clone(),
            density_bind_group_b: density_bind_group_b.clone(),
            integrate_bind_group_a_to_b: integrate_bind_group_a_to_b.clone(),
            integrate_bind_group_b_to_a: integrate_bind_group_b_to_a.clone(),

            grid_info,
            hash_assign_pipeline,
            bitonic_pipeline,
            cell_bounds_clear_pipeline,
            cell_bounds_pipeline,
            density_grid_pipeline,
            integrate_grid_pipeline,

            hash_assign_bind_group_a,
            hash_assign_bind_group_b,
            bitonic_bind_groups,
            cell_bounds_bind_group,
            density_grid_bind_group_a: density_bind_group_a,
            density_grid_bind_group_b: density_bind_group_b,
            integrate_grid_bind_group_a_to_b: integrate_bind_group_a_to_b,
            integrate_grid_bind_group_b_to_a: integrate_bind_group_b_to_a,
            grid_data_bind_group,

            _bitonic_stage_buffers: bitonic_stage_buffers,
        }
    }

    pub fn particle_count(&self) -> u32 {
        self.params.particle_count
    }

    pub fn using_grid(&self) -> bool {
        self.use_grid
    }

    pub fn toggle_neighbor_search(&mut self) {
        self.use_grid = !self.use_grid;
    }

    pub fn barricades(&self) -> &[Barricade] {
        &self.barricades
    }

    /// Resets the simulation to its initial dam-break state (same particle
    /// count/layout, since spacing/domain don't change at runtime) and
    /// clears all placed barricades. Triggered by the `R` key.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        let mut params = SimParams::new_dam_break(0, self.spacing);
        let particles = init_dam_break(&mut params, self.spacing);
        self.params = params;

        queue.write_buffer(&self.buffer_a, 0, bytemuck::cast_slice(&particles));
        queue.write_buffer(&self.buffer_b, 0, bytemuck::cast_slice(&particles));
        self.current_is_a = true;

        self.barricades.clear();
        queue.write_buffer(
            &self.barricade_count_buffer,
            0,
            bytemuck::bytes_of(&BarricadeListParams { count: 0, _pad: [0; 3] }),
        );
    }

    /// Places (or moves, if at capacity — oldest is dropped) a barricade at
    /// `center` and re-uploads the full barricade list. Cheap/infrequent
    /// (only called on user placement, not per frame).
    pub fn place_barricade(&mut self, queue: &wgpu::Queue, center: [f32; 3]) {
        use barricade::BARRICADE_HALF_EXTENTS;

        if self.barricades.len() >= MAX_BARRICADES {
            self.barricades.remove(0);
        }
        self.barricades.push(Barricade {
            center,
            _pad0: 0.0,
            half_extents: BARRICADE_HALF_EXTENTS,
            _pad1: 0.0,
        });

        queue.write_buffer(&self.barricade_buffer, 0, bytemuck::cast_slice(&self.barricades));
        queue.write_buffer(
            &self.barricade_count_buffer,
            0,
            bytemuck::bytes_of(&BarricadeListParams {
                count: self.barricades.len() as u32,
                _pad: [0; 3],
            }),
        );
    }

    /// The buffer holding the most recently integrated particle state —
    /// what the renderer should read this frame.
    pub fn current_buffer(&self) -> &wgpu::Buffer {
        if self.current_is_a {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    /// Advances the sim by `SUBSTEPS_PER_FRAME` fixed-`dt` substeps so simulated
    /// time keeps pace with wall-clock time at a stable small dt.
    pub fn step(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        for _ in 0..self.substeps_per_frame {
            self.substep(encoder);
        }
    }

    pub fn substeps_per_frame(&self) -> u32 {
        self.substeps_per_frame
    }

    /// Adjusts the simulation's time scale (substeps run per rendered
    /// frame): faster playback for demoing, or slower for inspecting a
    /// splash in more detail. Bound to `+`/`-`.
    pub fn adjust_time_scale(&mut self, delta: i32) {
        let new_value = self.substeps_per_frame as i32 + delta;
        self.substeps_per_frame = new_value.clamp(1, 32) as u32;
    }

    pub fn adaptive_dt_enabled(&self) -> bool {
        self.adaptive_dt
    }

    pub fn toggle_adaptive_dt(&mut self) {
        self.adaptive_dt = !self.adaptive_dt;
        if !self.adaptive_dt {
            self.params.dt = DT_MAX;
        }
    }

    /// Updates `dt` from the CFL condition given the current max particle
    /// speed (from the periodic `debug_stats` readback). No-op if adaptive
    /// dt is toggled off (fixed `DT_MAX` is used instead). Bound to `D`.
    pub fn update_cfl_dt(&mut self, max_speed: f32) {
        if !self.adaptive_dt {
            return;
        }
        let h = self.params.smoothing_radius;
        self.params.dt = (CFL_NUMBER * h / max_speed.max(1.0e-4)).clamp(DT_MIN, DT_MAX);
    }

    fn substep(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = self.params.particle_count.div_ceil(WORKGROUP_SIZE);

        if self.use_grid {
            let padded_workgroups = self.grid_info.padded_particle_count.div_ceil(WORKGROUP_SIZE);
            let cell_workgroups = self.grid_info.total_cells.div_ceil(WORKGROUP_SIZE);

            let hash_assign_bind_group = if self.current_is_a {
                &self.hash_assign_bind_group_a
            } else {
                &self.hash_assign_bind_group_b
            };
            let density_bind_group = if self.current_is_a {
                &self.density_grid_bind_group_a
            } else {
                &self.density_grid_bind_group_b
            };
            let integrate_bind_group = if self.current_is_a {
                &self.integrate_grid_bind_group_a_to_b
            } else {
                &self.integrate_grid_bind_group_b_to_a
            };

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sph-grid-pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.hash_assign_pipeline);
            pass.set_bind_group(0, hash_assign_bind_group, &[]);
            pass.dispatch_workgroups(padded_workgroups, 1, 1);

            pass.set_pipeline(&self.bitonic_pipeline);
            for bind_group in &self.bitonic_bind_groups {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(padded_workgroups, 1, 1);
            }

            pass.set_pipeline(&self.cell_bounds_clear_pipeline);
            pass.set_bind_group(0, &self.cell_bounds_bind_group, &[]);
            pass.dispatch_workgroups(cell_workgroups, 1, 1);

            pass.set_pipeline(&self.cell_bounds_pipeline);
            pass.set_bind_group(0, &self.cell_bounds_bind_group, &[]);
            pass.dispatch_workgroups(padded_workgroups, 1, 1);

            pass.set_pipeline(&self.density_grid_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            pass.set_bind_group(1, &self.grid_data_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);

            pass.set_pipeline(&self.integrate_grid_pipeline);
            pass.set_bind_group(0, integrate_bind_group, &[]);
            pass.set_bind_group(1, &self.grid_data_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        } else {
            let density_bind_group = if self.current_is_a {
                &self.density_bind_group_a
            } else {
                &self.density_bind_group_b
            };
            let integrate_bind_group = if self.current_is_a {
                &self.integrate_bind_group_a_to_b
            } else {
                &self.integrate_bind_group_b_to_a
            };

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sph-naive-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.density_pipeline);
            pass.set_bind_group(0, density_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);

            pass.set_pipeline(&self.integrate_pipeline);
            pass.set_bind_group(0, integrate_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.current_is_a = !self.current_is_a;
    }

    /// Blocking CPU readback of the current particle buffer for development
    /// diagnostics (NaN/blowup detection) — not part of the per-frame hot
    /// path, only called periodically from `App`'s debug stats log.
    pub fn debug_stats(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> ParticleStats {
        let buffer_size =
            (self.params.particle_count as usize * std::mem::size_of::<Particle>())
                as wgpu::BufferAddress;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug-stats-staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("debug-stats-copy"),
        });
        encoder.copy_buffer_to_buffer(self.current_buffer(), 0, &staging, 0, buffer_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        rx.recv()
            .expect("map_async callback never fired")
            .expect("failed to map staging buffer");

        let data = slice.get_mapped_range().expect("buffer not mapped");
        let particles: &[Particle] = bytemuck::cast_slice(&data);

        let mut stats = ParticleStats::default();
        for p in particles {
            let speed = (p.velocity[0] * p.velocity[0]
                + p.velocity[1] * p.velocity[1]
                + p.velocity[2] * p.velocity[2])
                .sqrt();
            if !p.position[1].is_finite() || !speed.is_finite() {
                stats.nan_count += 1;
                continue;
            }
            stats.min_y = stats.min_y.min(p.position[1]);
            stats.max_y = stats.max_y.max(p.position[1]);
            stats.max_speed = stats.max_speed.max(speed);
            stats.max_density = stats.max_density.max(p.density);
        }
        drop(data);
        staging.unmap();
        stats
    }
}

#[derive(Debug)]
pub struct ParticleStats {
    pub min_y: f32,
    pub max_y: f32,
    pub max_speed: f32,
    pub max_density: f32,
    pub nan_count: u32,
}

impl Default for ParticleStats {
    fn default() -> Self {
        Self {
            min_y: f32::INFINITY,
            max_y: f32::NEG_INFINITY,
            max_speed: 0.0,
            max_density: 0.0,
            nan_count: 0,
        }
    }
}
