use std::mem::offset_of;

use ash::vk;

#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: glam::Vec4,
    pub normal: glam::Vec4,
    // tangent: glam::Vec4,
    // bitangent: glam::Vec4,
    pub tex_coords: glam::Vec4,
}

impl Vertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription> {
        vec![
            vk::VertexInputBindingDescription::default()
                .stride(size_of::<Self>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
        ]
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .offset(offset_of!(Self, position) as u32)
                .format(vk::Format::R32G32B32A32_SFLOAT),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .offset(offset_of!(Self, normal) as u32)
                .format(vk::Format::R32G32B32A32_SFLOAT),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .offset(offset_of!(Self, tex_coords) as u32)
                .format(vk::Format::R32G32B32A32_SFLOAT),
        ]
    }
}