#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_debug_printf : enable
// #extension GL_KHR_vulkan_glsl: enable

struct ObjectData {
  mat4 transform;
};

struct CamData {
  vec4 pos;
  vec4 look_at;
  mat4 view_proj_mat;
};

struct PointLight{
  vec4 pos;
  vec4 color;
  vec4 props;
};

struct SceneData {
  CamData cam_data;
  // uvec4 counts;
  // PointLight point_lights[32];
};