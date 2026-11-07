// Translucent floor ring marking how far destruction has spread from the
// dam — a direct, in-scene visualization of the destruction-vs-distance
// flood-impact metric (src/sim/debris.rs), not just the settings-panel
// table/graph. Separate pipeline from background.wgsl because this is the
// only geometry in the scene that needs alpha blending.

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
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
