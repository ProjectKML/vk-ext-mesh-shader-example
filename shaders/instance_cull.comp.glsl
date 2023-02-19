#version 460

#include "types.glsl"

layout(set = 0, binding = 0) readonly buffer InstanceBuffer {
    Instance[] instances;
};

layout(set = 0, binding = 1) readonly buffer NumInstancesBuffer {
    uint num_instances;
};

layout(set = 0, binding = 2) buffer VisibleInstanceBuffer {
    VisibleInstance[] visible_instances;
};

layout(set = 0, binding = 3) buffer NumVisibleInstances {
    uint num_visible_instances;
};

void main() {

}