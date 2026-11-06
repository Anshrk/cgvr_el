pub mod point_sprite;

use crate::camera::{Camera, CameraUniform};
use crate::gpu::GpuContext;
use crate::sim::SphSim;
use point_sprite::PointSpritePipeline;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Owns the depth buffer and the render passes, and draws one frame.
///
/// M1 only runs the point-sprite debug pass; M3 adds the full SSFR pass
/// chain here alongside it (point-sprite stays as a toggleable debug view).
pub struct Renderer {
    depth_view: wgpu::TextureView,
    point_sprite: PointSpritePipeline,
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        let depth_view = create_depth_view(gpu);
        let point_sprite = PointSpritePipeline::new(&gpu.device, gpu.config.format, DEPTH_FORMAT);
        Self {
            depth_view,
            point_sprite,
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext) {
        self.depth_view = create_depth_view(gpu);
    }

    pub fn render_frame(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        sph: &SphSim,
        camera: &Camera,
    ) {
        let camera_uniform = CameraUniform::from_camera(camera, 0.08);
        self.point_sprite.update_camera(&gpu.queue, &camera_uniform);
        let bind_group = self
            .point_sprite
            .create_bind_group(&gpu.device, sph.current_buffer());

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.05,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.point_sprite
            .draw(&mut render_pass, &bind_group, sph.particle_count());
    }
}

fn create_depth_view(gpu: &GpuContext) -> wgpu::TextureView {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth-texture"),
        size: wgpu::Extent3d {
            width: gpu.config.width,
            height: gpu.config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
