#version 460

layout(location = 0) in vec2 tex_coords;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec4 out_color;

#include "utils.glsl"

void main() {
    const vec3 meshlet_color = murmur_hash_11_color(gl_PrimitiveID);

    out_color = vec4(meshlet_color, 1.0);
}