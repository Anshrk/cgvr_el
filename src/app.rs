use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

use crate::camera::{Camera, OrbitController};
use crate::gpu::GpuContext;
use crate::render::ui::EguiState;
use crate::render::Renderer;
use crate::sim::barricade::BARRICADE_HALF_EXTENTS;
use crate::sim::particle::DamBreakConfig;
use crate::sim::{ParticleStats, SphSim};

struct Scene {
    gpu: GpuContext,
    renderer: Renderer,
    sph: SphSim,
    camera: Camera,
    orbit: OrbitController,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    cursor_pos: (f64, f64),
    paused: bool,
    last_frame: Instant,
    last_stats_log: Instant,
    last_stats: ParticleStats,
    last_frame_ms: f32,
    frame_index: u64,
    egui_state: EguiState,
    ui_config: DamBreakConfig,
    show_settings: bool,
    /// The settings panel's on-screen rect (logical points, previous frame)
    /// — used to gate camera input ourselves instead of trusting egui's
    /// `consumed` flag, which is unreliable here: `egui::Context::
    /// is_pointer_over_egui` falls back to `true` *everywhere* whenever
    /// there's no central panel / root-level UI (we only use a floating
    /// `egui::Window`), which is exactly our case.
    settings_panel_rect: Option<egui::Rect>,
}

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    scene: Option<Scene>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("CGVR — Real-Time SPH Flood Simulation")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let gpu = pollster::block_on(GpuContext::new(window.clone()));
        let ui_config = DamBreakConfig::default();
        let sph = SphSim::new(&gpu.device, ui_config);
        let renderer = Renderer::new(&gpu, sph.params.domain_min, sph.params.domain_max);
        let egui_state = EguiState::new(&gpu.device, &window, gpu.config.format);

        let aspect = gpu.config.width as f32 / gpu.config.height.max(1) as f32;
        let camera = Camera {
            eye: glam::Vec3::new(0.0, 3.0, 10.0),
            target: glam::Vec3::new(0.0, 2.0, 0.0),
            up: glam::Vec3::Y,
            fov_y_radians: 60f32.to_radians(),
            aspect,
            znear: 0.05,
            zfar: 100.0,
        };
        let orbit = OrbitController::new(camera.target, 10.0);

        self.window = Some(window);
        self.scene = Some(Scene {
            gpu,
            renderer,
            sph,
            camera,
            orbit,
            dragging: false,
            last_cursor: None,
            cursor_pos: (0.0, 0.0),
            paused: false,
            last_frame: Instant::now(),
            last_stats_log: Instant::now(),
            last_stats: ParticleStats::default(),
            last_frame_ms: 0.0,
            frame_index: 0,
            egui_state,
            ui_config,
            show_settings: true,
            settings_panel_rect: None,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(window), Some(scene)) = (self.window.as_ref(), self.scene.as_mut()) else {
            return;
        };

        // NOT using the `consumed` flag from this: egui's `is_pointer_over_egui`
        // falls back to `true` *everywhere on screen* whenever there's no
        // central panel / root-level UI content (we only use a floating
        // `egui::Window`), which is exactly our case — so it's unusable for
        // gating camera input here. Still feed events through so widgets
        // (sliders, buttons) work; just don't trust the returned bool.
        let _ = scene.egui_state.handle_event(window, &event);
        let over_panel = |scene: &Scene, pos: (f64, f64)| -> bool {
            scene.settings_panel_rect.is_some_and(|rect| {
                let scale = window.scale_factor() as f32;
                rect.contains(egui::pos2(pos.0 as f32 / scale, pos.1 as f32 / scale))
            })
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                scene.gpu.resize(new_size.width, new_size.height);
                scene.renderer.resize(&scene.gpu);
                scene.camera.aspect =
                    scene.gpu.config.width as f32 / scene.gpu.config.height.max(1) as f32;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } if !over_panel(scene, scene.cursor_pos) => {
                scene.dragging = state == ElementState::Pressed;
                if !scene.dragging {
                    scene.last_cursor = None;
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } if !over_panel(scene, scene.cursor_pos) => {
                place_barricade_at_cursor(scene);
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Always track the cursor position (needed for barricade
                // placement), but only drive camera orbit when the cursor
                // isn't over the settings panel.
                scene.cursor_pos = (position.x, position.y);
                if scene.dragging && !over_panel(scene, scene.cursor_pos) {
                    if let Some((lx, ly)) = scene.last_cursor {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        scene.orbit.orbit(dx, dy);
                    }
                }
                scene.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } if !over_panel(scene, scene.cursor_pos) => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 40.0) as f32,
                };
                scene.orbit.zoom(amount);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let Key::Character(s) = &event.logical_key {
                        if s.as_str().eq_ignore_ascii_case("g") {
                            scene.sph.toggle_neighbor_search();
                            eprintln!(
                                "[cgvr] neighbor search: {}",
                                if scene.sph.using_grid() { "grid (M2)" } else { "naive (M1)" }
                            );
                        }
                        if s.as_str().eq_ignore_ascii_case("f") {
                            scene.renderer.toggle_ssfr();
                            eprintln!(
                                "[cgvr] render view: {}",
                                if scene.renderer.ssfr_enabled() { "SSFR (M3)" } else { "point-sprite debug (M1)" }
                            );
                        }
                        if s.as_str().eq_ignore_ascii_case("r") {
                            scene.sph.reset(&scene.gpu.queue);
                            scene.renderer.rebuild_barricades(&scene.gpu.device, scene.sph.barricades());
                            eprintln!("[cgvr] simulation reset");
                        }
                        if s.as_str().eq_ignore_ascii_case("p") {
                            scene.paused = !scene.paused;
                            eprintln!("[cgvr] {}", if scene.paused { "paused" } else { "resumed" });
                        }
                        if s.as_str().eq_ignore_ascii_case("q") {
                            event_loop.exit();
                        }
                        if s.as_str().eq_ignore_ascii_case("d") {
                            scene.sph.toggle_adaptive_dt();
                            eprintln!(
                                "[cgvr] adaptive CFL dt: {}",
                                if scene.sph.adaptive_dt_enabled() { "on" } else { "off (fixed dt)" }
                            );
                        }
                        if s.as_str().eq_ignore_ascii_case("v") {
                            scene.renderer.toggle_vorticity_heatmap();
                            eprintln!(
                                "[cgvr] vorticity heatmap: {}",
                                if scene.renderer.vorticity_heatmap_enabled() { "on" } else { "off" }
                            );
                        }
                        if s.as_str().eq_ignore_ascii_case("c") {
                            scene.orbit.reset();
                            eprintln!("[cgvr] camera reset");
                        }
                        if s.as_str() == "+" || s.as_str() == "=" {
                            scene.sph.adjust_time_scale(1);
                            eprintln!("[cgvr] time scale: {} substeps/frame", scene.sph.substeps_per_frame());
                        }
                        if s.as_str() == "-" {
                            scene.sph.adjust_time_scale(-1);
                            eprintln!("[cgvr] time scale: {} substeps/frame", scene.sph.substeps_per_frame());
                        }
                        if s.as_str().eq_ignore_ascii_case("u") {
                            scene.show_settings = !scene.show_settings;
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                step_and_render(scene, window);
                window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Right-click placement: casts a ray from the camera through the cursor
/// and intersects it with the domain's floor plane, standing in for 6-DOF
/// hand placement in a real VR integration (see camera.rs's `screen_ray`).
fn place_barricade_at_cursor(scene: &mut Scene) {
    let (width, height) = (scene.gpu.config.width as f32, scene.gpu.config.height.max(1) as f32);
    let ndc_x = (scene.cursor_pos.0 as f32 / width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (scene.cursor_pos.1 as f32 / height) * 2.0;

    let (origin, dir) = scene.camera.screen_ray(ndc_x, ndc_y);
    let floor_y = scene.sph.params.domain_min[1];
    if dir.y.abs() < 1.0e-6 {
        return;
    }
    let t = (floor_y - origin.y) / dir.y;
    if t <= 0.0 {
        return;
    }
    let hit = origin + dir * t;

    let center = [hit.x, floor_y + BARRICADE_HALF_EXTENTS[1], hit.z];
    scene.sph.place_barricade(&scene.gpu.queue, center);
    scene.renderer.rebuild_barricades(&scene.gpu.device, scene.sph.barricades());
    eprintln!("[cgvr] barricade placed at ({:.2}, {:.2}, {:.2})", center[0], center[1], center[2]);
}

/// Draws the settings panel: live profiling readouts, the keybind
/// reference, and editable dam-break config with a "Rebuild" button
/// (particle-count changes need a full `SphSim` rebuild — new buffer
/// sizes, new grid dims — so they can't be a live drag-slider like the
/// camera or physics toggles). Returns true if "Rebuild" was clicked.
fn draw_settings_panel(
    scene: &mut Scene,
    window: &Window,
    color_view: &wgpu::TextureView,
    encoder: &mut wgpu::CommandEncoder,
) -> bool {
    let fps = 1000.0 / scene.last_frame_ms.max(0.001);
    let frame_ms = scene.last_frame_ms;
    let particle_count = scene.sph.particle_count();
    let barricade_count = scene.sph.barricades().len();
    let using_grid = scene.sph.using_grid();
    let ssfr_enabled = scene.renderer.ssfr_enabled();
    let vorticity_on = scene.renderer.vorticity_heatmap_enabled();
    let adaptive_dt = scene.sph.adaptive_dt_enabled();
    let dt = scene.sph.params.dt;
    let substeps = scene.sph.substeps_per_frame();
    let paused = scene.paused;
    let stats = scene.last_stats;

    let debris_total = scene.sph.debris().len();
    let swept_count = scene
        .sph
        .debris()
        .iter()
        .filter(|d| d.displacement() > crate::sim::debris::SWEPT_THRESHOLD)
        .count();
    let max_displacement = scene
        .sph
        .debris()
        .iter()
        .map(|d| d.displacement())
        .fold(0.0_f32, f32::max);

    let gpu = &scene.gpu;
    let ui_config = &mut scene.ui_config;
    let mut rebuild = false;
    let mut panel_rect = None;

    scene.egui_state.render(
        &gpu.device,
        &gpu.queue,
        encoder,
        window,
        color_view,
        gpu.config.width,
        gpu.config.height,
        |ctx| {
            let window_response = egui::Window::new("CGVR Settings (U to hide)")
                .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.set_max_width(260.0);

                    egui::CollapsingHeader::new("Status").default_open(true).show(ui, |ui| {
                        ui.label(format!("FPS: {fps:.0} ({frame_ms:.2} ms/frame)"));
                        ui.label(format!("Particles: {particle_count}"));
                        ui.label(format!("Barricades: {barricade_count} / 8"));
                        ui.label(format!(
                            "Neighbor search: {}",
                            if using_grid { "grid (M2)" } else { "naive O(n^2) (M1)" }
                        ));
                        ui.label(format!(
                            "Render view: {}",
                            if ssfr_enabled { "SSFR (M3)" } else { "point-sprite debug (M1)" }
                        ));
                        ui.label(format!("Vorticity heatmap: {}", if vorticity_on { "on" } else { "off" }));
                        ui.label(format!(
                            "Timestep: {dt:.5} ({})",
                            if adaptive_dt { "adaptive CFL" } else { "fixed" }
                        ));
                        ui.label(format!("Time scale: {substeps} substeps/frame"));
                        ui.label(format!("State: {}", if paused { "paused" } else { "running" }));
                        ui.label(format!(
                            "Last reading: y=[{:.2},{:.2}] max_speed={:.2} max_density={:.1} nan={}",
                            stats.min_y, stats.max_y, stats.max_speed, stats.max_density, stats.nan_count
                        ));
                    });

                    egui::CollapsingHeader::new("Flood Impact").default_open(true).show(ui, |ui| {
                        ui.label(format!("Debris swept away: {swept_count} / {debris_total}"));
                        ui.label(format!("Max displacement: {max_displacement:.2} m"));
                    });

                    egui::CollapsingHeader::new("Dam-break config").default_open(true).show(ui, |ui| {
                        ui.add(egui::Slider::new(&mut ui_config.spacing, 0.06..=0.25).text("particle spacing"));
                        ui.add(egui::Slider::new(&mut ui_config.width_frac, 0.1..=0.6).text("dam width fraction"));
                        ui.add(egui::Slider::new(&mut ui_config.height_frac, 0.1..=0.9).text("dam height fraction"));
                        ui.label("(smaller spacing = more particles — rebuild to apply)");
                        if ui.button("Rebuild simulation").clicked() {
                            rebuild = true;
                        }
                    });

                    egui::CollapsingHeader::new("Keybinds").default_open(false).show(ui, |ui| {
                        egui::Grid::new("keybind-grid").striped(true).show(ui, |ui| {
                            let rows: &[(&str, &str)] = &[
                                ("Left-drag", "Orbit camera"),
                                ("Scroll", "Zoom"),
                                ("Right-click", "Place barricade"),
                                ("G", "Toggle neighbor search (grid/naive)"),
                                ("F", "Toggle render view (SSFR/debug)"),
                                ("V", "Toggle vorticity heatmap"),
                                ("D", "Toggle adaptive CFL timestep"),
                                ("R", "Reset simulation"),
                                ("P", "Pause/resume"),
                                ("C", "Reset camera"),
                                ("+ / -", "Speed up / slow down time scale"),
                                ("U", "Toggle this panel"),
                                ("Q", "Quit"),
                            ];
                            for (key, action) in rows {
                                ui.label(*key);
                                ui.label(*action);
                                ui.end_row();
                            }
                        });
                    });
                });

            if let Some(response) = window_response {
                panel_rect = Some(response.response.rect);
            }
        },
    );

    scene.settings_panel_rect = panel_rect;
    rebuild
}

fn step_and_render(scene: &mut Scene, window: &Window) {
    scene.orbit.apply(&mut scene.camera);

    let surface_texture = match scene.gpu.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(tex) => tex,
        wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
        wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            scene.gpu.surface.configure(&scene.gpu.device, &scene.gpu.config);
            return;
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            eprintln!("wgpu surface validation error");
            return;
        }
    };

    let color_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = scene
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame-encoder"),
        });

    if !scene.paused {
        scene.sph.step(&scene.gpu.queue, &mut encoder);
    }
    scene
        .renderer
        .render_frame(&scene.gpu, &mut encoder, &color_view, &scene.sph, &scene.camera);

    let rebuild_requested = if scene.show_settings {
        Some(draw_settings_panel(scene, window, &color_view, &mut encoder))
    } else {
        scene.settings_panel_rect = None;
        None
    };

    scene.gpu.queue.submit(Some(encoder.finish()));
    scene.gpu.queue.present(surface_texture);

    // Gated on !paused, same as sph.step(): when paused, step() (and the
    // accumulator clear inside it) never ran this frame, so reading it back
    // here would just re-apply the same stale impulse repeatedly.
    if !scene.paused {
        scene.sph.update_debris_from_gpu(&scene.gpu.device, &scene.gpu.queue);
    }
    scene.renderer.update_debris(&scene.gpu.queue, scene.sph.debris());

    if rebuild_requested == Some(true) {
        scene.sph = SphSim::new(&scene.gpu.device, scene.ui_config);
        scene.renderer.rebuild_barricades(&scene.gpu.device, &[]);
        eprintln!(
            "[cgvr] simulation rebuilt: {} particles (spacing={:.3}, width_frac={:.2}, height_frac={:.2})",
            scene.sph.particle_count(),
            scene.ui_config.spacing,
            scene.ui_config.width_frac,
            scene.ui_config.height_frac,
        );
    }

    scene.frame_index += 1;
    let now = Instant::now();
    let frame_ms = (now - scene.last_frame).as_secs_f32() * 1000.0;
    scene.last_frame = now;
    scene.last_frame_ms = frame_ms;
    let early_frame = scene.frame_index <= 30 && scene.frame_index % 2 == 0;
    if early_frame || (now - scene.last_stats_log).as_secs_f32() > 1.0 {
        scene.last_stats_log = now;
        let stats = scene.sph.debug_stats(&scene.gpu.device, &scene.gpu.queue);
        scene.sph.update_cfl_dt(stats.max_speed);
        eprintln!(
            "[cgvr] particles={} frame={:.2}ms (~{:.0} FPS) y=[{:.2},{:.2}] max_speed={:.2} max_density={:.1} dt={:.5} nan={}",
            scene.sph.particle_count(),
            frame_ms,
            1000.0 / frame_ms.max(0.001),
            stats.min_y,
            stats.max_y,
            stats.max_speed,
            stats.max_density,
            scene.sph.params.dt,
            stats.nan_count,
        );
        scene.last_stats = stats;

        let max_disp = scene.sph.debris().iter().map(|d| d.displacement()).fold(0.0_f32, f32::max);
        let positions: Vec<String> = scene
            .sph
            .debris()
            .iter()
            .map(|d| format!("({:.2},{:.2},{:.2})", d.gpu.center[0], d.gpu.center[1], d.gpu.center[2]))
            .collect();
        eprintln!("[cgvr] debris max_displacement={max_disp:.3} positions={positions:?}");
    }
}
