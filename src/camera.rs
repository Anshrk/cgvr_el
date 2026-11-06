use glam::{Mat4, Vec3};

/// A simple look-at/perspective camera plus the per-frame uniform data the
/// point-sprite and (later) SSFR shaders need for billboarding.
///
/// M1 only needs a single mono view for inspecting the simulation; M4 adds
/// a `CameraSource`/`EyeView` abstraction on top of this for stereo rendering
/// without changing this struct's math.
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aspect: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn view_proj(&self) -> Mat4 {
        let view = glam::camera::rh::view::look_at_mat4(self.eye, self.target, self.up);
        // wgpu's NDC is Z in [0, 1], Y-up, matching glam's "directx" convention bucket.
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fov_y_radians,
            self.aspect,
            self.znear,
            self.zfar,
        );
        proj * view
    }

    /// Camera-space right/up basis vectors, used to billboard particle quads
    /// so they always face the viewer.
    pub fn right_up(&self) -> (Vec3, Vec3) {
        let forward = (self.target - self.eye).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        (right, up)
    }
}

/// Mouse-orbit + scroll-zoom controller around a fixed target point.
pub struct OrbitController {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitController {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.35,
            distance,
        }
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.005;
        self.yaw += dx * SENSITIVITY;
        self.pitch = (self.pitch - dy * SENSITIVITY).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.5).clamp(1.0, 50.0);
    }

    pub fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.cos();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.sin();
        self.target + Vec3::new(x, y, z)
    }

    pub fn apply(&self, camera: &mut Camera) {
        camera.eye = self.eye();
        camera.target = self.target;
    }
}

/// Mirrors `CameraUniform` in `shaders/point_sprite.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_right: [f32; 3],
    pub particle_radius: f32,
    pub camera_up: [f32; 3],
    pub _pad: f32,
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera, particle_radius: f32) -> Self {
        let (right, up) = camera.right_up();
        Self {
            view_proj: camera.view_proj().to_cols_array_2d(),
            camera_right: right.to_array(),
            particle_radius,
            camera_up: up.to_array(),
            _pad: 0.0,
        }
    }
}
