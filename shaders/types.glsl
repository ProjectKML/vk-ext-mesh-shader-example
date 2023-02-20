struct Globals {
    mat4 view_projection_matrix;
    vec4 frustum_planes[6];
    vec3 camera_pos;
    float time;
};

struct Vertex {
    float position_x, position_y, position_z;
    float tex_coord_x, tex_coord_y;
    float normal_x, normal_y, normal_z;
};

struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

struct Meshlet {
    AABB aabb;
    uint data_offset;
    uint vertex_count;
    uint triangle_count;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer VertexRef {
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
    uint num_meshlets;
};

layout(buffer_reference, std430, buffer_reference_align = 4) buffer MeshLevelRef {
    MeshLevel value;
};

struct Mesh {
    MeshLevelRef levels;
    uint num_levels;
};

struct Instance {
    float position_x, position_y, position_z;
    float scale;
    float rotation_x, rotation_y, rotation_z, rotation_w;
    uint mesh_idx;
};

struct VisibleInstance {
    uint index;
    uint level;
};