// M3 SSFR stage 4/4: combine the smoothed depth + reconstructed normal into
// the final shaded water color, composited over whatever the background
// pass already drew (this pass discards wherever there's no fluid, so the
// background shows through unmodified at those pixels). Fresnel/refraction:
// `background_grab_tex` is a copy of the opaque background taken right
// before this pass runs (see `SsfrPipeline::grab_background`), sampled at a
// normal-distorted UV for refraction; a cheap analytic sky gradient stands
// in for a reflection source (no real environment map in this scene).

struct GBufferParams {
    tan_half_fov_y: f32,
    aspect: f32,
    resolution: vec2<f32>,
    empty_sentinel: f32,
    show_vorticity: f32,
    vorticity_scale: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: GBufferParams;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var vorticity_tex: texture_2d<f32>;
@group(0) @binding(4) var background_grab_tex: texture_2d<f32>;
@group(0) @binding(5) var background_sampler: sampler;

// How strongly the surface normal bends the refraction UV — tuned
// empirically, same as every other constant in this project; too high
// smears the background into an unrecognizable blur, too low looks flat.
const REFRACTION_STRENGTH: f32 = 0.025;

// Cheap stand-in for a real environment/reflection map: a vertical gradient
// from a horizon tone up to a brighter "sky" tone, keyed by the normal's
// world-space-ish up component (view-space Y, since normal_tex stores
// view-space normals — close enough for a glint, not meant to be physical).
fn sky_color(normal: vec3<f32>) -> vec3<f32> {
    let horizon = vec3<f32>(0.5, 0.65, 0.75);
    let zenith = vec3<f32>(0.85, 0.95, 1.0);
    let up = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(horizon, zenith, up);
}

// Simple blue -> cyan -> yellow -> red heatmap ramp.
fn heatmap(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let c0 = vec3<f32>(0.05, 0.05, 0.6);
    let c1 = vec3<f32>(0.1, 0.8, 0.9);
    let c2 = vec3<f32>(0.95, 0.9, 0.1);
    let c3 = vec3<f32>(0.9, 0.1, 0.1);
    if (t < 0.33) {
        return mix(c0, c1, t / 0.33);
    } else if (t < 0.66) {
        return mix(c1, c2, (t - 0.33) / 0.33);
    } else {
        return mix(c2, c3, (t - 0.66) / 0.34);
    }
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, vertex_index == 1u);
    let y = select(-1.0, 3.0, vertex_index == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(floor(frag_pos.xy));
    let depth = textureLoad(depth_tex, coord, 0).r;
    if (depth >= params.empty_sentinel) {
        discard;
    }

    let normal = textureLoad(normal_tex, coord, 0).xyz;
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);

    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.6));
    let diffuse = max(dot(normal, light_dir), 0.0);

    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0);

    let deep_water = vec3<f32>(0.04, 0.2, 0.45);
    let shallow_water = vec3<f32>(0.25, 0.6, 0.85);
    let base_color = mix(deep_water, shallow_water, diffuse);

    // Refraction: a distorted glimpse of whatever's behind the fluid
    // (ground/debris/damage ring), tinted with the water's own color so it
    // still reads as colored water rather than a clear window.
    let uv = frag_pos.xy / params.resolution;
    let refract_uv = clamp(uv + normal.xy * REFRACTION_STRENGTH, vec2<f32>(0.001), vec2<f32>(0.999));
    let refraction_color = textureSampleLevel(background_grab_tex, background_sampler, refract_uv, 0.0).rgb;
    let tinted_refraction = mix(refraction_color, base_color, 0.55);

    // Reflection: no real environment map, so an analytic sky gradient
    // stands in, blended in more strongly at grazing angles (fresnel).
    let reflection_color = sky_color(normal);
    var color = mix(tinted_refraction, reflection_color, fresnel * 0.6) * (0.35 + 0.65 * diffuse);

    if (params.show_vorticity > 0.5) {
        let vorticity = textureLoad(vorticity_tex, coord, 0).r;
        let t = vorticity / max(params.vorticity_scale, 1.0e-4);
        color = mix(color, heatmap(t), 0.75);
    }

    return vec4<f32>(color, 1.0);
}
