#version 460

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_shader_8bit_storage : require

layout(local_size_x = 32) in;
layout(max_vertices = 64, max_primitives = 126, triangles) out;

layout(location = 0) out vec2[] out_tex_coords;
layout(location = 1) out vec3[] out_normals;
layout(location = 2) out vec3[] out_colors;

layout(push_constant) uniform PushConstants {
    mat4 view_projection_matrix;
} push_constants;

struct Vertex {
    float position_x, position_y, position_z;
    float tex_coord_x, tex_coord_y;
    float normal_x, normal_y, normal_z;
};

layout(set = 0, binding = 0) readonly buffer Vertices {
    Vertex[] vertices;
};

struct Meshlet {
    uint data_offset;
    uint vertex_count;
    uint triangle_count;
};

layout(set = 0, binding = 1) readonly buffer Meshlets {
    Meshlet[] meshlets;
};

layout(set = 0, binding = 2) readonly buffer MeshletData {
    uint[] meshlet_data;
};

layout(set = 0, binding = 2) readonly buffer ByteMeshletData {
    uint8_t[] byte_meshlet_data;
};

uint murmur_hash_11(uint src) {
    const uint M = 0x5bd1e995;
    uint h = 1190494759;
    src *= M;
    src ^= src >> 24;
    src *= M;
    h *= M;
    h ^= src;
    h ^= h >> 13;
    h *= M;
    h ^= h >> 15;

    return h;
}

vec3 murmur_hash_11_color(uint src) {
    const uint hash = murmur_hash_11(src);
    return vec3(float((hash >> 16) & 0xFF), float((hash >> 8) & 0xFF), float(hash & 0xFF)) / 256.0;
}

uint get_index(uint index_offset, uint index) {
    const uint byte_offset = (3 - (index & 3)) << 3;
    return (meshlet_data[index_offset + (index >> 2)] & (0xFF << byte_offset)) >> byte_offset;
}

void main() {
    const uint liid = gl_LocalInvocationIndex;
    const uint meshlet_idx = gl_WorkGroupID.x;

    const Meshlet meshlet = meshlets[meshlet_idx];
    SetMeshOutputsEXT(meshlet.vertex_count, meshlet.triangle_count);

    const vec3 meshlet_color = murmur_hash_11_color(meshlet_idx);

    for(uint i = liid; i < meshlet.vertex_count; i += 32) {
        const uint vertex_idx = meshlet_data[meshlet.data_offset + i];
        const Vertex vertex = vertices[vertex_idx];

        gl_MeshVerticesEXT[i].gl_Position = push_constants.view_projection_matrix * vec4(vertex.position_x, vertex.position_y, vertex.position_z, 1.0);

        out_tex_coords[i] = vec2(vertex.tex_coord_x, vertex.tex_coord_y);
        out_normals[i] = vec3(vertex.normal_x, vertex.normal_y, vertex.normal_z);
        out_colors[i] = meshlet_color;
    }

    const uint index_offset = meshlet.data_offset + meshlet.vertex_count;

    for(uint i = liid; i < meshlet.triangle_count; i += 32) {
        const uint triangle_idx = 3 * i;
        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(get_index(index_offset, triangle_idx), get_index(index_offset,  triangle_idx + 1), get_index(index_offset, triangle_idx + 2));
    }
}