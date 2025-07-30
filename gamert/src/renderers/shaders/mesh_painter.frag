#version 460 core

#include "mesh_painter_common.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inUV;
layout (location = 2) flat in uint objId;

layout (location = 0) out vec4 outFragColor;

layout(std430, set = 0, binding = 2) buffer readonly ssbo3 { ObjectInfo object_buffer [];};
layout(set = 1, binding = 0) uniform sampler samplers[1];
layout(set = 2, binding = 0) uniform texture2D textures[];

void main() {
    outFragColor = texture(sampler2D(textures[object_buffer[objId].tex_id], samplers[0]), inUV);
    //outFragColor = vec4(1.0,1.0,1.0,1.0);
}
