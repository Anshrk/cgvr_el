use wgpu::util::DeviceExt;

use crate::camera::CameraUniform;
use crate::sim::barricade::Barricade;
use crate::sim::debris::{DebrisState, MAX_DEBRIS};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BackgroundVertex {
    position: [f32; 3],
    color: [f32; 3],
}

fn quad(corners: [[f32; 3]; 4], color: [f32; 3]) -> [BackgroundVertex; 6] {
    let [a, b, c, d] = corners;
    [
        BackgroundVertex { position: a, color },
        BackgroundVertex { position: b, color },
        BackgroundVertex { position: c, color },
        BackgroundVertex { position: a, color },
        BackgroundVertex { position: c, color },
        BackgroundVertex { position: d, color },
    ]
}

/// Builds the static tank geometry (ground plane + 4 boundary walls) that
/// visually frames the simulation domain. Not the same thing as M5's
/// user-placed barricades — this is fixed scene dressing so the dam-break
/// has visible surroundings instead of floating in void, needed from M3 on
/// both as a depth-test reference for the fluid and (M6) a refraction source.
fn build_tank_geometry(domain_min: [f32; 3], domain_max: [f32; 3]) -> Vec<BackgroundVertex> {
    let margin = (domain_max[0] - domain_min[0]) * 0.15;
    let ground_color = [0.22, 0.24, 0.27];
    let wall_color = [0.45, 0.5, 0.56];

    let gx0 = domain_min[0] - margin;
    let gx1 = domain_max[0] + margin;
    let gz0 = domain_min[2] - margin;
    let gz1 = domain_max[2] + margin;
    let gy = domain_min[1];

    let mut vertices = Vec::new();

    // Ground plane.
    vertices.extend(quad(
        [
            [gx0, gy, gz0],
            [gx1, gy, gz0],
            [gx1, gy, gz1],
            [gx0, gy, gz1],
        ],
        ground_color,
    ));

    let (x0, x1, z0, z1, y0, y1) = (
        domain_min[0], domain_max[0], domain_min[2], domain_max[2], domain_min[1], domain_max[1],
    );

    // -X wall
    vertices.extend(quad(
        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
        wall_color,
    ));
    // +X wall
    vertices.extend(quad(
        [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
        wall_color,
    ));
    // -Z wall
    vertices.extend(quad(
        [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        wall_color,
    ));
    // +Z wall
    vertices.extend(quad(
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        wall_color,
    ));

    vertices
}

/// Generates all 6 faces of an axis-aligned box. No backface culling is
/// configured on this pipeline, so winding order doesn't matter here.
fn box_quads(center: [f32; 3], half: [f32; 3], color: [f32; 3]) -> Vec<BackgroundVertex> {
    let (x0, x1) = (center[0] - half[0], center[0] + half[0]);
    let (y0, y1) = (center[1] - half[1], center[1] + half[1]);
    let (z0, z1) = (center[2] - half[2], center[2] + half[2]);

    let mut vertices = Vec::with_capacity(36);
    vertices.extend(quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], color)); // -X
    vertices.extend(quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], color)); // +X
    vertices.extend(quad([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], color)); // +Y
    vertices.extend(quad([[x0, y0, z1], [x1, y0, z1], [x1, y0, z0], [x0, y0, z0]], color)); // -Y
    vertices.extend(quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], color)); // -Z
    vertices.extend(quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], color)); // +Z
    vertices
}

const BARRICADE_COLOR: [f32; 3] = [0.75, 0.35, 0.2];
const DEBRIS_VERTS_PER_OBJECT: usize = 36;

pub struct BackgroundPipeline {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    barricade_vertex_buffer: Option<wgpu::Buffer>,
    barricade_vertex_count: u32,
    // Debris moves every frame (unlike barricades, which only change on
    // placement), so this is a fixed-capacity buffer re-written in place via
    // `queue.write_buffer` each frame instead of reallocated.
    debris_vertex_buffer: wgpu::Buffer,
    debris_vertex_count: u32,
}

impl BackgroundPipeline {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        domain_min: [f32; 3],
        domain_max: [f32; 3],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("background"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/background.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("background-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("background-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BackgroundVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("background-camera-uniform"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                camera_right: [1.0, 0.0, 0.0],
                particle_radius: 0.08,
                camera_up: [0.0, 1.0, 0.0],
                _pad: 0.0,
                view: glam::Mat4::IDENTITY.to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let vertices = build_tank_geometry(domain_min, domain_max);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("background-vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_count = vertices.len() as u32;

        // The bind group never needs to change frame-to-frame (no ping-pong
        // buffer here, unlike the particle passes), so build it once.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("background-bind-group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let debris_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debris-vertices"),
            size: (MAX_DEBRIS * DEBRIS_VERTS_PER_OBJECT * std::mem::size_of::<BackgroundVertex>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            camera_buffer,
            bind_group,
            vertex_buffer,
            vertex_count,
            barricade_vertex_buffer: None,
            barricade_vertex_count: 0,
            debris_vertex_buffer,
            debris_vertex_count: 0,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, uniform: &CameraUniform) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniform));
    }

    /// Rebuilds the barricade box geometry. Called only when the list
    /// actually changes (on placement), not every frame.
    pub fn rebuild_barricades(&mut self, device: &wgpu::Device, barricades: &[Barricade]) {
        let mut vertices = Vec::with_capacity(barricades.len() * 36);
        for b in barricades {
            vertices.extend(box_quads(b.center, b.half_extents, BARRICADE_COLOR));
        }
        self.barricade_vertex_count = vertices.len() as u32;
        self.barricade_vertex_buffer = if vertices.is_empty() {
            None
        } else {
            Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("barricade-vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }))
        };
    }

    /// Re-writes the (fixed-capacity) debris vertex buffer in place. Called
    /// every frame since debris moves continuously, unlike barricades.
    pub fn update_debris(
        &mut self,
        queue: &wgpu::Queue,
        debris: &[DebrisState],
        thresholds: &crate::sim::debris::ForceThresholds,
    ) {
        let mut vertices = Vec::with_capacity(debris.len() * DEBRIS_VERTS_PER_OBJECT);
        for d in debris {
            vertices.extend(box_quads(d.gpu.center, d.gpu.half_extents, d.display_color(thresholds)));
        }
        self.debris_vertex_count = vertices.len() as u32;
        if !vertices.is_empty() {
            queue.write_buffer(&self.debris_vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }
    }

    pub fn draw<'pass>(&'pass self, render_pass: &mut wgpu::RenderPass<'pass>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);

        if let Some(buf) = &self.barricade_vertex_buffer {
            render_pass.set_vertex_buffer(0, buf.slice(..));
            render_pass.draw(0..self.barricade_vertex_count, 0..1);
        }

        if self.debris_vertex_count > 0 {
            render_pass.set_vertex_buffer(0, self.debris_vertex_buffer.slice(..));
            render_pass.draw(0..self.debris_vertex_count, 0..1);
        }
    }
}
