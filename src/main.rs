mod app;
mod camera;
mod gpu;
mod render;
mod sim;

use winit::event_loop::{ControlFlow, EventLoop};

use app::App;

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).expect("event loop error");
}
