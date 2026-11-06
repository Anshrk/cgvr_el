use winit::event::WindowEvent;
use winit::window::Window;

/// Owns the egui context/input-state/renderer trio and draws one UI pass per
/// frame, composited on top of whatever the scene passes already drew
/// (`LoadOp::Load`). Houses the settings panel: live profiling readouts,
/// the keybind reference, and editable dam-break config with a rebuild
/// button (particle-count changes need a full GPU buffer reallocation, so
/// they can't be a live drag-slider like the physics constants are).
pub struct EguiState {
    ctx: egui::Context,
    winit_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl EguiState {
    pub fn new(device: &wgpu::Device, window: &Window, color_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(device, color_format, egui_wgpu::RendererOptions::default());

        Self {
            ctx,
            winit_state,
            renderer,
        }
    }

    /// Feeds a window event to egui; returns true if egui consumed it (the
    /// caller should skip its own handling of that event, e.g. camera
    /// orbit, when this is true — so clicking/dragging over the panel
    /// doesn't also spin the camera underneath it).
    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.winit_state.on_window_event(window, event).consumed
    }

    pub fn render<R>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        color_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        run_ui: impl FnOnce(&egui::Context) -> R,
    ) -> R {
        let raw_input = self.winit_state.take_egui_input(window);
        self.ctx.begin_pass(raw_input);
        let ui_result = run_ui(&self.ctx);
        let full_output = self.ctx.end_pass();
        self.winit_state
            .handle_platform_output(window, full_output.platform_output.clone());

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = self.ctx.tessellate(full_output.shapes, pixels_per_point);

        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point,
        };

        self.renderer
            .update_buffers(device, queue, encoder, &paint_jobs, &screen_descriptor);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui-pass"),
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
            let mut pass = pass.forget_lifetime();
            self.renderer.render(&mut pass, &paint_jobs, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        ui_result
    }
}
