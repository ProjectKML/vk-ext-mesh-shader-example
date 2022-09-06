#version 460

layout(location = 0) in vec2 tex_coords;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(1.0);
}