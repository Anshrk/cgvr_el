use wgpu::util::DeviceExt;

use crate::camera::CameraUniform;

/// M1 debug render pass: flat-shaded, camera-facing billboard quads read
/// straight out of the current particle storage buffer (no vertex buffer).
pub struct PointSpritePipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    camera_buffer: wgpu::Buffer,
}

impl PointSpritePipeline {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("point_sprite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_sprite.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point-sprite-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point-sprite-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point-sprite-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
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
            label: Some("camera-uniform"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                camera_right: [1.0, 0.0, 0.0],
                particle_radius: 0.08,
                camera_up: [0.0, 1.0, 0.0],
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            camera_buffer,
        }
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, uniform: &CameraUniform) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(uniform));
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        particle_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point-sprite-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn draw<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        bind_group: &'pass wgpu::BindGroup,
        particle_count: u32,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..6, 0..particle_count);
    }
}
