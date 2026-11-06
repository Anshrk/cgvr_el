// M3: opaque background geometry (ground plane + barricade boxes) rendered
// before the fluid. Needed from M3 onward both as the depth-test reference
// for the water and, later (M6), as the refraction source texture for
// ssfr_shade.wgsl.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_right: vec3<f32>,
    particle_radius: f32,
    camera_up: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_y: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_y = in.position.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Cheap height-based shading so the ground plane and barricades read as
    // distinct geometry without needing real lighting yet.
    let shade = clamp(0.6 + in.world_y * 0.05, 0.3, 1.0);
    return vec4<f32>(in.color * shade, 1.0);
}
