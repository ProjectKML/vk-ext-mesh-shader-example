#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 32) in;
layout(max_vertices = 64, max_primitives = 124, triangles) out;

layout(location = 0) out vec2[] out_tex_coords;
layout(location = 1) out vec3[] out_normals;
layout(location = 2) out vec3[] out_colors;

struct Vertex {
    float position_x, position_y, position_z;
    float tex_coord_x, tex_coord_y;
    float normal_x, normal_y, normal_z;
};

struct Meshlet {
    uint data_offset;
    uint vertex_count;
    uint triangle_count;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer VertexRef {
    Vertex value;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer MeshletRef {
    Meshlet value;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer MeshletDataRef {
    uint value;
};

struct MeshLevel {
    VertexRef vertices;
    MeshletRef meshlets;
    MeshletDataRef meshlet_data;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer MeshLevelRef {
    MeshLevel value;
};

struct Mesh {
    MeshLevelRef levels;
    uint num_levels;
};

layout(set = 0, binding = 0) uniform Constants {
	mat4 view_projection_matrix;
	vec3 camera_pos;
} constants;

layout(set = 0, binding = 1) readonly buffer MeshBuffersBuffer {
    Mesh meshes[];
};

layout(push_constant) uniform PushConstants {
    float translation_x, translation_y, translation_z, scale;
	vec4 rotation;
    uint mesh_idx;
    uint level_idx;
} push_constants;

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

uint get_index(MeshletDataRef meshlet_data, uint index_offset, uint index) {
    const uint byte_offset = (3 - (index & 3)) << 3;
    return (meshlet_data[index_offset + (index >> 2)].value & (0xFF << byte_offset)) >> byte_offset;
}

vec4 calculate_pos(mat4 view_projection_matrix, vec3 position, vec3 translation, float scale, vec4 rotation) {
	vec3 translated_pos = scale * position + translation;
	vec3 target_pos = translated_pos + 2.0 * cross(rotation.xyz, cross(rotation.xyz, translated_pos) + rotation.w * translated_pos);
	return view_projection_matrix * vec4(target_pos, 1.0);
}

void main() {
    const uint liid = gl_LocalInvocationIndex;
    const uint meshlet_idx = gl_WorkGroupID.x;

    MeshLevel mesh_level = meshes[push_constants.mesh_idx].levels[push_constants.level_idx].value;

    const Meshlet meshlet = mesh_level.meshlets[meshlet_idx].value;
    SetMeshOutputsEXT(meshlet.vertex_count, meshlet.triangle_count);

    const vec3 meshlet_color = murmur_hash_11_color(meshlet_idx);

    MeshletDataRef meshlet_data = mesh_level.meshlet_data;

    for(uint i = liid; i < meshlet.vertex_count; i += 32) {
        const uint vertex_idx = meshlet_data[meshlet.data_offset + i].value;
        const Vertex vertex = mesh_level.vertices[vertex_idx].value;

        gl_MeshVerticesEXT[i].gl_Position = calculate_pos(constants.view_projection_matrix, 
			vec3(vertex.position_x, vertex.position_y, vertex.position_z), 
			vec3(push_constants.translation_x, push_constants.translation_y, push_constants.translation_z), push_constants.scale, push_constants.rotation);

        out_tex_coords[i] = vec2(vertex.tex_coord_x, vertex.tex_coord_y);
        out_normals[i] = vec3(vertex.normal_x, vertex.normal_y, vertex.normal_z);
        out_colors[i] = meshlet_color;
    }

    const uint index_offset = meshlet.data_offset + meshlet.vertex_count;

    for(uint i = liid; i < meshlet.triangle_count; i += 32) {
        const uint triangle_idx = 3 * i;
        gl_PrimitiveTriangleIndicesEXT[i] = uvec3(get_index(meshlet_data, index_offset, triangle_idx),
            get_index(meshlet_data, index_offset,  triangle_idx + 1),
            get_index(meshlet_data, index_offset, triangle_idx + 2));
    }
}