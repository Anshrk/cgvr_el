// M1 debug rendering: camera-facing billboard quads with a fake analytic
// sphere normal, flat-lit. Replaced (visually) by the full screen-space
// fluid rendering pipeline in M3 — this pass stays around as a cheap
// debug/validation view (e.g. for the M1 vs M2 neighbor-search comparison).

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
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) quad_uv: vec2<f32>,
    @location(1) speed: f32,
};

// Unit quad corners in [-1, 1], two triangles via a 6-vertex fan.
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
    out.speed = length(particle.velocity);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0) {
        discard;
    }

    // Analytic sphere normal from the quad-local disc, cheap stand-in for
    // real per-pixel depth reconstruction (that's what M3's SSFR pipeline
    // does properly from an actual depth buffer).
    let z = sqrt(1.0 - r2);
    let normal = normalize(vec3<f32>(in.quad_uv, z));

    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.5));
    let diffuse = max(dot(normal, light_dir), 0.0);

    let water_color = vec3<f32>(0.15, 0.45, 0.85);
    let speed_tint = clamp(in.speed * 0.05, 0.0, 1.0);
    let color = mix(water_color, vec3<f32>(0.8, 0.9, 1.0), speed_tint) * (0.3 + 0.7 * diffuse);

    return vec4<f32>(color, 1.0);
}
