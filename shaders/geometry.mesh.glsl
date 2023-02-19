#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_mesh_shader : require

layout(local_size_x = LOCAL_SIZE_X) in;
layout(max_vertices = 64, max_primitives = 124, triangles) out;

#include "types.glsl"
#include "utils.glsl"

layout(location = 0) out vec2[] out_tex_coords;
layout(location = 1) out vec3[] out_normals;
layout(location = 2) out vec3[] out_colors;

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

    const vec3 meshlet_color = murmur_hash_11_color(meshlet_idx ^ floatBitsToInt(length(vec3(push_constants.translation_x, push_constants.translation_y, push_constants.translation_z))));

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