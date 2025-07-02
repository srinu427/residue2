use std::sync::Arc;

use ash::vk;

use crate::{Image2d, Painter};

#[derive(Debug, Clone, Copy)]
pub enum ShaderInputType {
    UniformBuffer,
    StorageBuffer,
    SampledImage2d,
    Sampler,
}

impl ShaderInputType {
    pub fn get_descriptor_type(&self) -> vk::DescriptorType {
        match self {
            ShaderInputType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            ShaderInputType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            ShaderInputType::SampledImage2d => vk::DescriptorType::SAMPLED_IMAGE,
            ShaderInputType::Sampler => vk::DescriptorType::SAMPLER,
        }
    }
}

pub enum ShaderInputValue {
    UniformBuffers(Vec<vk::Buffer>),
    StorageBuffers(Vec<vk::Buffer>),
    SampledImage2ds(Vec<vk::ImageView>),
    Samplers(Vec<vk::Sampler>),
}

#[derive(Debug, Clone, Copy)]
pub struct ShaderInputBindingInfo {
    pub _type: ShaderInputType,
    pub count: u32,
}

pub struct ShaderInputLayout {
    bindings: Vec<ShaderInputBindingInfo>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    painter: Arc<Painter>,
}

impl ShaderInputLayout {
    pub fn new(painter: Arc<Painter>, bindings: Vec<ShaderInputBindingInfo>) -> Result<Self, String> {
        unsafe {
            let vk_bindings = bindings
                .iter()
                .enumerate()
                .map(|(binding, binding_info)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding as u32)
                        .descriptor_type(binding_info._type.get_descriptor_type())
                        .descriptor_count(binding_info.count)
                        .stage_flags(vk::ShaderStageFlags::ALL)
                })
                .collect::<Vec<_>>();
            let descriptor_set_layout = painter
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings),
                    None,
                )
                .map_err(|e| format!("at descriptor set layout creation: {e}"))?;
            Ok(Self { descriptor_set_layout, bindings, painter: painter.clone() })
        }
    }
}

impl Drop for ShaderInputLayout {
    fn drop(&mut self) {
        unsafe {
            self
                .painter
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct ShaderInputAllocator {
    painter: Arc<Painter>,
    descriptor_pool: vk::DescriptorPool,
}

impl ShaderInputAllocator {
    pub fn new(painter: Arc<Painter>, counts: Vec<(ShaderInputType, u32)>, max_sets: u32) -> Result<Self, String> {
        let pool_sizes = counts
            .iter()
            .map(|(ty, count)| vk::DescriptorPoolSize::default().ty(ty.get_descriptor_type()).descriptor_count(*count))
            .collect::<Vec<_>>();
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            painter
                .device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(|e| format!("at descriptor pool creation: {e}"))?
        };
        Ok(Self { painter, descriptor_pool })
    }

    pub fn allocate(&self, layouts: Vec<&ShaderInputLayout>) -> Result<Vec<ShaderInput>, String> {
        let set_layouts = layouts
            .iter()
            .map(|layout| layout.descriptor_set_layout)
            .collect::<Vec<_>>();
        let shader_inputs = unsafe {
            self.painter.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(|e| format!("at descriptor set allocation: {e}"))?
                .iter()
                .map(|vk_dset| ShaderInput { descriptor_set: *vk_dset, descriptor_pool: self.descriptor_pool, is_freeable: true, painter: self.painter.clone() })
                .collect::<Vec<_>>()
        };
        Ok(shader_inputs)
    }
}

impl Drop for ShaderInputAllocator {
    fn drop(&mut self) {
        unsafe {
            self
                .painter
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

pub struct ShaderInput {
    pub descriptor_set: vk::DescriptorSet,
    pub descriptor_pool: vk::DescriptorPool,
    pub is_freeable: bool,
    painter: Arc<Painter>,
}

impl ShaderInput {
    pub fn write(&self, bindings: Vec<ShaderInputValue>) {
        let uniform_buffer_infos = bindings
            .iter()
            .enumerate()
            .map(|(binding, binding_value)| {
                match binding_value {
                    ShaderInputValue::UniformBuffers(buffers) => {
                        Some(vk::DescriptorBufferInfo::default()
                            .buffer(*buffers)
                            .offset(0)
                            .range(vk::WHOLE_SIZE))
                    }
                    _ => None
                }
            })
            .collect::<Vec<_>>();
        let writes = bindings
            .iter()
            .enumerate()
            .map(|(binding, binding_value)| {
                match binding_value {
                    ShaderInputValue::UniformBuffers(buffers) => {
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(binding as u32)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(
                                buffers
                                    .iter()
                                    .map(|buffer| {
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(*buffer)
                                            .offset(0)
                                            .range(vk::WHOLE_SIZE)
                                    })
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            )
                    }
                    ShaderInputValue::StorageBuffers(buffers) => {
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(binding as u32)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(
                                buffers
                                    .iter()
                                    .map(|buffer| {
                                        vk::DescriptorBufferInfo::default()
                                            .buffer(*buffer)
                                            .offset(0)
                                            .range(vk::WHOLE_SIZE)
                                    })
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            )
                    }
                    ShaderInputValue::SampledImage2ds(images) => {
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(binding as u32)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(
                                images
                                    .iter()
                                    .map(|image| {
                                        vk::DescriptorImageInfo::default()
                                            .image_view(*image)
                                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                    })
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            )
                    }
                    ShaderInputValue::Samplers(samplers) => {
                        vk::WriteDescriptorSet::default()
                            .dst_set(self.descriptor_set)
                            .dst_binding(binding as u32)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(
                                samplers
                                    .iter()
                                    .map(|sampler| {
                                        vk::DescriptorImageInfo::default()
                                            .sampler(*sampler)
                                    })
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            )
                    }
                }
            })
            .collect::<Vec<_>>();
        unsafe {
            self.painter.device.update_descriptor_sets(&writes, &[]);
        }
    }
}
