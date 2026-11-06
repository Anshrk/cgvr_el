use wgpu::util::DeviceExt;

use super::depth_splat::{DEPTH_TARGET_FORMAT, EMPTY_SENTINEL};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    direction: [i32; 2],
    radius: i32,
    sigma_spatial: f32,
    sigma_range: f32,
    empty_sentinel: f32,
    _pad0: f32,
    _pad1: f32,
}

/// SSFR stage 2: separable bilateral blur. One shared pipeline, run twice
/// per frame (horizontal then vertical) against different input textures
/// and a different static direction-params buffer each time.
pub struct BilateralBlurPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    horizontal_params: wgpu::Buffer,
    vertical_params: wgpu::Buffer,
}

impl BilateralBlurPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bilateral_blur"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bilateral_blur.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bilateral-blur-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bilateral-blur-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bilateral-blur-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DEPTH_TARGET_FORMAT,
                    blend: None,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let radius = 6;
        let sigma_spatial = 3.0_f32;
        let sigma_range = 0.08_f32;

        let horizontal_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur-params-horizontal"),
            contents: bytemuck::bytes_of(&BlurParams {
                direction: [1, 0],
                radius,
                sigma_spatial,
                sigma_range,
                empty_sentinel: EMPTY_SENTINEL,
                _pad0: 0.0,
                _pad1: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let vertical_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur-params-vertical"),
            contents: bytemuck::bytes_of(&BlurParams {
                direction: [0, 1],
                radius,
                sigma_spatial,
                sigma_range,
                empty_sentinel: EMPTY_SENTINEL,
                _pad0: 0.0,
                _pad1: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        Self {
            pipeline,
            bind_group_layout,
            horizontal_params,
            vertical_params,
        }
    }

    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        input_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bilateral-blur-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input_view) },
            ],
        })
    }

    pub fn horizontal_bind_group(&self, device: &wgpu::Device, input_view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.create_bind_group(device, &self.horizontal_params, input_view)
    }

    pub fn vertical_bind_group(&self, device: &wgpu::Device, input_view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.create_bind_group(device, &self.vertical_params, input_view)
    }

    pub fn run<'pass>(
        &'pass self,
        render_pass: &mut wgpu::RenderPass<'pass>,
        bind_group: &'pass wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
