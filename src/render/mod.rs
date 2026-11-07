pub mod background;
pub mod point_sprite;
pub mod ssfr;
pub mod ui;

use crate::camera::{Camera, CameraUniform};
use crate::gpu::GpuContext;
use crate::sim::SphSim;
use background::BackgroundPipeline;
use point_sprite::PointSpritePipeline;
use ssfr::SsfrPipeline;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Owns the depth buffer and the render passes, and draws one frame.
///
/// The background/tank pass always runs first into a shared depth buffer.
/// After that, either the M1 point-sprite debug view or the M3 SSFR pass
/// chain runs (toggleable via `toggle_ssfr` — SSFR is new and unverified
/// visually, so it defaults off until confirmed to look right).
pub struct Renderer {
    depth_view: wgpu::TextureView,
    background: BackgroundPipeline,
    point_sprite: PointSpritePipeline,
    ssfr: SsfrPipeline,
    ssfr_enabled: bool,
}

impl Renderer {
    pub fn new(gpu: &GpuContext, domain_min: [f32; 3], domain_max: [f32; 3]) -> Self {
        let depth_view = create_depth_view(gpu);
        let background = BackgroundPipeline::new(
            &gpu.device,
            gpu.config.format,
            DEPTH_FORMAT,
            domain_min,
            domain_max,
        );
        let point_sprite = PointSpritePipeline::new(&gpu.device, gpu.config.format, DEPTH_FORMAT);
        let ssfr = SsfrPipeline::new(
            &gpu.device,
            gpu.config.format,
            DEPTH_FORMAT,
            gpu.config.width,
            gpu.config.height,
        );
        Self {
            depth_view,
            background,
            point_sprite,
            ssfr,
            ssfr_enabled: false,
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext) {
        self.depth_view = create_depth_view(gpu);
        self.ssfr.resize(&gpu.device, gpu.config.width, gpu.config.height);
    }

    pub fn toggle_ssfr(&mut self) {
        self.ssfr_enabled = !self.ssfr_enabled;
    }

    pub fn ssfr_enabled(&self) -> bool {
        self.ssfr_enabled
    }

    pub fn rebuild_barricades(&mut self, device: &wgpu::Device, barricades: &[crate::sim::barricade::Barricade]) {
        self.background.rebuild_barricades(device, barricades);
    }

    pub fn update_debris(&mut self, queue: &wgpu::Queue, debris: &[crate::sim::debris::DebrisState]) {
        self.background.update_debris(queue, debris);
    }

    pub fn toggle_vorticity_heatmap(&mut self) {
        self.ssfr.toggle_vorticity_heatmap();
    }

    pub fn vorticity_heatmap_enabled(&self) -> bool {
        self.ssfr.vorticity_heatmap_enabled()
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
        self.background.update_camera(&gpu.queue, &camera_uniform);

        // Background pass: always first, establishes the depth buffer that
        // either the debug or SSFR pass below tests/loads against.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("background-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.05, b: 0.1, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.background.draw(&mut pass);
        }

        if self.ssfr_enabled {
            self.ssfr.render(
                &gpu.device,
                &gpu.queue,
                encoder,
                color_view,
                &self.depth_view,
                sph,
                camera,
            );
        } else {
            self.point_sprite.update_camera(&gpu.queue, &camera_uniform);
            let bind_group = self
                .point_sprite
                .create_bind_group(&gpu.device, sph.current_buffer());

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("point-sprite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.point_sprite.draw(&mut pass, &bind_group, sph.particle_count());
        }
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
