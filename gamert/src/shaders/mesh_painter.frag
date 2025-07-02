#version 460

#include "mesh_painter_common.glsl"

layout (location = 0) in vec4 inPosition;
layout (location = 1) in vec4 inNormal;
layout (location = 2) in vec4 inUV;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler samplers[1];
layout(set = 1, binding = 1) uniform texture2D textures[];

layout(push_constant) uniform ObjInfo {
    uint obj_id;
    uint mesh_id;
    uint texture_id;
} obj_info;

void main() {
    //outFragColor = texture(sampler2D(textures[obj_info.texture_id], samplers[0]), inUV.xy);
    outFragColor = vec4(1.0,1.0,1.0,1.0);
}