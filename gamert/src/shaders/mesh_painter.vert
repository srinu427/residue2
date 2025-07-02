#version 460

#include "mesh_painter_common.glsl"

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inNormal;
layout(location = 2) in vec4 inUV;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outUV;

layout(std140, set = 0, binding = 0) buffer readonly ScnWrap { SceneData data;} scene_data;

vec4 invert_y_axis(vec4 v) {
    return vec4(v.x, -v.y, v.z, v.w);
}

void main() {
    outPosition = inPosition;
    outNormal = inNormal;
    outUV = inUV;
    gl_Position = invert_y_axis(scene_data.data.cam_data.view_proj_mat * inPosition);
    // debugPrintfEXT("My vec is %v", gl_Position);
}