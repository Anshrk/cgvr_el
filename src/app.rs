use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

use crate::camera::{Camera, OrbitController};
use crate::gpu::GpuContext;
use crate::render::Renderer;
use crate::sim::barricade::BARRICADE_HALF_EXTENTS;
use crate::sim::SphSim;

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
    frame_index: u64,
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
        let sph = SphSim::new(&gpu.device);
        let renderer = Renderer::new(&gpu, sph.params.domain_min, sph.params.domain_max);

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
            frame_index: 0,
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
            } => {
                scene.dragging = state == ElementState::Pressed;
                if !scene.dragging {
                    scene.last_cursor = None;
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                place_barricade_at_cursor(scene);
            }
            WindowEvent::CursorMoved { position, .. } => {
                scene.cursor_pos = (position.x, position.y);
                if scene.dragging {
                    if let Some((lx, ly)) = scene.last_cursor {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        scene.orbit.orbit(dx, dy);
                    }
                }
                scene.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
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
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                step_and_render(scene);
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

fn step_and_render(scene: &mut Scene) {
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

    scene.gpu.queue.submit(Some(encoder.finish()));
    scene.gpu.queue.present(surface_texture);

    scene.frame_index += 1;
    let now = Instant::now();
    let frame_ms = (now - scene.last_frame).as_secs_f32() * 1000.0;
    scene.last_frame = now;
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
    }
}
