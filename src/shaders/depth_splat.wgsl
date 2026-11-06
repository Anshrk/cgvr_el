// M3 SSFR stage 1/4: render each particle as a camera-facing billboard,
// reconstructing an analytic sphere depth profile per-pixel (not just a
// flat disc) so the later bilateral-blur + normal-reconstruction stages
// have a depth field that actually looks like overlapping fluid blobs
// rather than a "ball pit" of flat circles.
//
// Writes linear view-space depth to a dedicated R32Float color target
// (read by bilateral_blur.wgsl / normals_from_depth.wgsl), while also
// using a real depth attachment for the hardware depth test so closest-
// particle-wins occlusion works between overlapping sprites and against
// the background. The hardware depth test uses the flat quad's own
// interpolated depth (good enough: within one small camera-facing quad
// the true depth variation is tiny), while depth_out carries the precise
// analytic value the later passes actually need.

struct Particle {
    position: vec3<f32>,
    density: f32,
    velocity: vec3<f32>,
    pressure: f32,
    vorticity: f32,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_right: vec3<f32>,
    particle_radius: f32,
    camera_up: vec3<f32>,
    _pad: f32,
    view: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) view_center_z: f32,
    // M6: carried straight through to the shading pass (unblurred — it's a
    // color-coding scalar, not geometry) for the vorticity heatmap.
    @location(2) vorticity: f32,
};

struct FragmentOutput {
    @location(0) depth_out: f32,
    @location(1) vorticity_out: f32,
};

const QUAD_CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let particle = particles[instance_index];
    let corner = QUAD_CORNERS[vertex_index];

    let world_offset = (camera.camera_right * corner.x + camera.camera_up * corner.y)
        * camera.particle_radius;
    let world_position = particle.position + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.quad_uv = corner;
    // glam's right-handed view space looks down -Z, so distance-from-camera
    // is the negated view-space Z of the (un-offset) particle center.
    let view_center = camera.view * vec4<f32>(particle.position, 1.0);
    out.view_center_z = -view_center.z;
    out.vorticity = particle.vorticity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }

    // Analytic sphere profile: pixels nearer the sprite's center are
    // nearer the camera by sqrt(1 - r^2) * radius.
    let z_offset = sqrt(1.0 - r2) * camera.particle_radius;

    var out: FragmentOutput;
    out.depth_out = in.view_center_z - z_offset;
    out.vorticity_out = in.vorticity;
    return out;
}
