pub mod bilateral_blur;
pub mod depth_splat;
pub mod normals;
pub mod shading;

use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraUniform};
use crate::sim::SphSim;
use bilateral_blur::BilateralBlurPipeline;
use depth_splat::{DepthSplatPipeline, DEPTH_TARGET_FORMAT, EMPTY_SENTINEL, VORTICITY_TARGET_FORMAT};
use normals::{NormalsPipeline, NORMAL_TARGET_FORMAT};
use shading::ShadingPipeline;

/// Mirrors `GBufferParams` in `normals_from_depth.wgsl` / `ssfr_shade.wgsl`
/// (the latter also reads `show_vorticity`/`vorticity_scale`; the former
/// only reads the common prefix up through `empty_sentinel`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GBufferParams {
    pub tan_half_fov_y: f32,
    pub aspect: f32,
    pub resolution: [f32; 2],
    pub empty_sentinel: f32,
    pub show_vorticity: f32,
    pub vorticity_scale: f32,
    pub _pad2: f32,
}

/// Empirically chosen so the heatmap's mid-range lands on the vorticity
/// magnitudes actually seen near barricade corners/edges in this sim's
/// parameter regime — not a universal constant.
const DEFAULT_VORTICITY_SCALE: f32 = 15.0;

struct Targets {
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
    blur_tmp_tex: wgpu::Texture,
    blur_tmp_view: wgpu::TextureView,
    normal_tex: wgpu::Texture,
    normal_view: wgpu::TextureView,
    vorticity_tex: wgpu::Texture,
    vorticity_view: wgpu::TextureView,
    // A copy of the opaque background (ground/barricades/debris/damage
    // ring), grabbed right before the shading pass overwrites those pixels
    // with the fluid's color — the refraction source for ssfr_shade.wgsl.
    // Same format as the swapchain (copy_texture_to_texture requires it).
    background_grab_tex: wgpu::Texture,
    background_grab_view: wgpu::TextureView,
}

fn create_targets(device: &wgpu::Device, color_format: wgpu::TextureFormat, width: u32, height: u32) -> Targets {
    let make = |label: &str, format: wgpu::TextureFormat, extra_usage: wgpu::TextureUsages| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | extra_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };

    let (depth_tex, depth_view) = make("ssfr-depth", DEPTH_TARGET_FORMAT, wgpu::TextureUsages::empty());
    let (blur_tmp_tex, blur_tmp_view) = make("ssfr-blur-tmp", DEPTH_TARGET_FORMAT, wgpu::TextureUsages::empty());
    let (normal_tex, normal_view) = make("ssfr-normal", NORMAL_TARGET_FORMAT, wgpu::TextureUsages::empty());
    let (vorticity_tex, vorticity_view) = make("ssfr-vorticity", VORTICITY_TARGET_FORMAT, wgpu::TextureUsages::empty());
    let (background_grab_tex, background_grab_view) =
        make("ssfr-background-grab", color_format, wgpu::TextureUsages::COPY_DST);

    Targets {
        depth_tex,
        depth_view,
        blur_tmp_tex,
        blur_tmp_view,
        normal_tex,
        normal_view,
        vorticity_tex,
        vorticity_view,
        background_grab_tex,
        background_grab_view,
    }
}

/// Orchestrates the four screen-space fluid rendering passes (depth splat
/// -> separable bilateral blur -> normal reconstruction -> shading). Draws
/// into whatever color/depth attachments the caller set up for the frame;
/// callers are expected to have already run the opaque background pass
/// against the same depth attachment with `LoadOp::Load` carried forward
/// into the depth-splat pass, and to keep using `LoadOp::Load` for the
/// final shading pass so non-fluid pixels keep the background's color.
pub struct SsfrPipeline {
    depth_splat: DepthSplatPipeline,
    bilateral_blur: BilateralBlurPipeline,
    normals: NormalsPipeline,
    shading: ShadingPipeline,

    gbuffer_params: wgpu::Buffer,
    targets: Targets,
    show_vorticity: bool,
    color_format: wgpu::TextureFormat,
}

impl SsfrPipeline {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let depth_splat = DepthSplatPipeline::new(device, depth_format);
        let bilateral_blur = BilateralBlurPipeline::new(device);
        let normals = NormalsPipeline::new(device);
        let shading = ShadingPipeline::new(device, color_format);

        let gbuffer_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gbuffer-params"),
            contents: bytemuck::bytes_of(&GBufferParams {
                tan_half_fov_y: 1.0,
                aspect: 1.0,
                resolution: [width as f32, height as f32],
                empty_sentinel: EMPTY_SENTINEL,
                show_vorticity: 0.0,
                vorticity_scale: DEFAULT_VORTICITY_SCALE,
                _pad2: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let targets = create_targets(device, color_format, width, height);

        Self {
            depth_splat,
            bilateral_blur,
            normals,
            shading,
            gbuffer_params,
            targets,
            show_vorticity: false,
            color_format,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.targets = create_targets(device, self.color_format, width, height);
    }

    /// Copies the already-drawn opaque background into `background_grab_tex`
    /// before the shading pass overwrites those pixels with the fluid's
    /// color — the refraction source for `ssfr_shade.wgsl`. Must run after
    /// the background pass and before `render`. No-op if the sizes don't
    /// match (e.g. a resize landed between passes this frame).
    pub fn grab_background(&self, encoder: &mut wgpu::CommandEncoder, swapchain_texture: &wgpu::Texture) {
        if swapchain_texture.size() != self.targets.background_grab_tex.size() {
            return;
        }
        encoder.copy_texture_to_texture(
            swapchain_texture.as_image_copy(),
            self.targets.background_grab_tex.as_image_copy(),
            self.targets.background_grab_tex.size(),
        );
    }

    pub fn toggle_vorticity_heatmap(&mut self) {
        self.show_vorticity = !self.show_vorticity;
    }

    pub fn vorticity_heatmap_enabled(&self) -> bool {
        self.show_vorticity
    }

    /// Runs the full depth-splat -> blur -> normals -> shading chain.
    /// `shared_depth_view` must be the same depth attachment the preceding
    /// background pass drew into, loaded (not cleared), so particle/
    /// background occlusion is correct.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        shared_depth_view: &wgpu::TextureView,
        sph: &SphSim,
        camera: &Camera,
    ) {
        let camera_uniform = CameraUniform::from_camera(camera, 0.08);
        self.depth_splat.update_camera(queue, &camera_uniform);

        let gbuffer = GBufferParams {
            tan_half_fov_y: (camera.fov_y_radians * 0.5).tan(),
            aspect: camera.aspect,
            resolution: [
                self.targets.depth_tex.width() as f32,
                self.targets.depth_tex.height() as f32,
            ],
            empty_sentinel: EMPTY_SENTINEL,
            show_vorticity: if self.show_vorticity { 1.0 } else { 0.0 },
            vorticity_scale: DEFAULT_VORTICITY_SCALE,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.gbuffer_params, 0, bytemuck::bytes_of(&gbuffer));

        // --- stage 1: depth splat (depth-tests against the shared depth buffer) ---
        let particle_bind_group = self.depth_splat.create_bind_group(device, sph.current_buffer());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssfr-depth-splat"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.depth_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: EMPTY_SENTINEL as f64, g: 0.0, b: 0.0, a: 0.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.targets.vorticity_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shared_depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.depth_splat.draw(&mut pass, &particle_bind_group, sph.particle_count());
        }

        // --- stage 2: separable bilateral blur (horizontal, then vertical) ---
        let h_bind_group = self.bilateral_blur.horizontal_bind_group(device, &self.targets.depth_view);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssfr-blur-horizontal"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.blur_tmp_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.bilateral_blur.run(&mut pass, &h_bind_group);
        }
        let v_bind_group = self.bilateral_blur.vertical_bind_group(device, &self.targets.blur_tmp_view);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssfr-blur-vertical"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.depth_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.bilateral_blur.run(&mut pass, &v_bind_group);
        }

        // --- stage 3: normal reconstruction from the now-smoothed depth ---
        let normals_bind_group = self.normals.create_bind_group(device, &self.gbuffer_params, &self.targets.depth_view);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssfr-normals"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.normals.run(&mut pass, &normals_bind_group);
        }

        // --- stage 4: shading, composited over the background (Load, not Clear) ---
        let shading_bind_group = self.shading.create_bind_group(
            device,
            &self.gbuffer_params,
            &self.targets.depth_view,
            &self.targets.normal_view,
            &self.targets.vorticity_view,
            &self.targets.background_grab_view,
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssfr-shade"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.shading.run(&mut pass, &shading_bind_group);
        }
    }
}
