// M3 SSFR stage 3/4: reconstruct a view-space surface normal per pixel from
// the (now bilateral-smoothed) depth buffer, via central differences on the
// reconstructed view-space positions of neighboring texels — not just a
// normal derived from the raw per-sprite UV (which depth_splat/point_sprite
// use as a cheap stand-in); this is the real SSFR normal reconstruction
// (van der Laan et al., 2009).

struct GBufferParams {
    tan_half_fov_y: f32,
    aspect: f32,
    resolution: vec2<f32>,
    empty_sentinel: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: GBufferParams;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, vertex_index == 1u);
    let y = select(-1.0, 3.0, vertex_index == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// Reconstructs a view-space position (camera looks down -Z) from a pixel's
// integer texel coord and its linear view-space depth (positive distance).
fn reconstruct_view_pos(coord: vec2<i32>, depth: f32) -> vec3<f32> {
    let ndc_x = (f32(coord.x) + 0.5) / params.resolution.x * 2.0 - 1.0;
    let ndc_y = 1.0 - (f32(coord.y) + 0.5) / params.resolution.y * 2.0;
    let view_x = ndc_x * params.tan_half_fov_y * params.aspect * depth;
    let view_y = ndc_y * params.tan_half_fov_y * depth;
    return vec3<f32>(view_x, view_y, -depth);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(floor(frag_pos.xy));
    let center_depth = textureLoad(depth_tex, coord, 0).r;
    if (center_depth >= params.empty_sentinel) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let dx_coord = coord + vec2<i32>(1, 0);
    let dy_coord = coord + vec2<i32>(0, 1);
    var depth_dx = textureLoad(depth_tex, dx_coord, 0).r;
    var depth_dy = textureLoad(depth_tex, dy_coord, 0).r;
    // Fall back to the center depth at silhouette edges rather than letting
    // a background/sentinel sample pull the reconstructed normal wildly off.
    if (depth_dx >= params.empty_sentinel) {
        depth_dx = center_depth;
    }
    if (depth_dy >= params.empty_sentinel) {
        depth_dy = center_depth;
    }

    let center_pos = reconstruct_view_pos(coord, center_depth);
    let dx_pos = reconstruct_view_pos(dx_coord, depth_dx);
    let dy_pos = reconstruct_view_pos(dy_coord, depth_dy);

    var normal = normalize(cross(dx_pos - center_pos, dy_pos - center_pos));
    if (normal.z < 0.0) {
        normal = -normal;
    }

    return vec4<f32>(normal, 1.0);
}
