#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

#include "types.glsl"

layout(set = 0, binding = 0) uniform GlobalsBuffer {
    Globals globals;
};

layout(set = 1, binding = 0) readonly buffer MeshBuffersBuffer {
    Mesh meshes[];
};

layout(set = 2, binding = 0) readonly buffer InstanceBuffer {
    Instance[] instances;
};

layout(set = 2, binding = 1) readonly buffer NumInstancesBuffer {
    uint num_instances;
};

layout(set = 2, binding = 2) buffer VisibleInstanceBuffer {
    VisibleInstance[] visible_instances;
};

layout(set = 2, binding = 3) buffer NumVisibleInstances {
    uint num_visible_instances;
};

void main() {
    const uint giid = gl_GlobalInvocationID.x;
    if(giid.x >= num_instances) {
        return;
    }


}