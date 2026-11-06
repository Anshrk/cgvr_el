// M3 SSFR stage 2/4: separable (horizontal then vertical) depth-range-
// weighted blur. This is the single biggest visual improvement in the SSFR
// chain — it's what turns the "ball pit" of overlapping sphere depths from
// depth_splat.wgsl into a single smooth fluid surface, while the range
// (depth-difference) weight keeps it from blurring across silhouette edges
// (e.g. the near and far side of the water shouldn't bleed into each other).
//
// Pixels with no particle (sentinel depth from depth_splat's clear value)
// are left untouched and excluded from neighboring pixels' weighted sums.

struct BlurParams {
    direction: vec2<i32>,
    radius: i32,
    sigma_spatial: f32,
    sigma_range: f32,
    empty_sentinel: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var depth_in: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, vertex_index == 1u);
    let y = select(-1.0, 3.0, vertex_index == 2u);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) f32 {
    let coord = vec2<i32>(floor(frag_pos.xy));
    let dims = vec2<i32>(textureDimensions(depth_in));
    let center = textureLoad(depth_in, coord, 0).r;

    if (center >= params.empty_sentinel) {
        return center;
    }

    var sum = 0.0;
    var weight_sum = 0.0;

    for (var i: i32 = -params.radius; i <= params.radius; i = i + 1) {
        let sample_coord = coord + params.direction * i;
        if (sample_coord.x < 0 || sample_coord.y < 0 || sample_coord.x >= dims.x || sample_coord.y >= dims.y) {
            continue;
        }
        let sample_depth = textureLoad(depth_in, sample_coord, 0).r;
        if (sample_depth >= params.empty_sentinel) {
            continue;
        }

        let spatial_w = exp(-f32(i * i) / (2.0 * params.sigma_spatial * params.sigma_spatial));
        let range_diff = sample_depth - center;
        let range_w = exp(-(range_diff * range_diff) / (2.0 * params.sigma_range * params.sigma_range));
        let w = spatial_w * range_w;

        sum = sum + sample_depth * w;
        weight_sum = weight_sum + w;
    }

    if (weight_sum <= 1.0e-4) {
        return center;
    }
    return sum / weight_sum;
}
