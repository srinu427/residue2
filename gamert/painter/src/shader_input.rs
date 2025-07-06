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
    pub dynamic: bool,
}

pub struct ShaderInputLayout {
    bindings: Vec<ShaderInputBindingInfo>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    painter: Arc<Painter>,
}

impl ShaderInputLayout {
    pub fn new(
        painter: Arc<Painter>,
        bindings: Vec<ShaderInputBindingInfo>,
    ) -> Result<Self, String> {
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

            let binding_flags = bindings
                .iter()
                .map(|binding_info| {
                    if binding_info.dynamic {
                        vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    } else {
                        vk::DescriptorBindingFlags::empty()
                    }
                })
                .collect::<Vec<_>>();

            let descriptor_set_layout = painter
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&vk_bindings)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&binding_flags),
                        ),
                    None,
                )
                .map_err(|e| format!("at descriptor set layout creation: {e}"))?;
            Ok(Self {
                descriptor_set_layout,
                bindings,
                painter: painter.clone(),
            })
        }
    }
}

impl Drop for ShaderInputLayout {
    fn drop(&mut self) {
        unsafe {
            self.painter
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
    pub fn new(
        painter: Arc<Painter>,
        counts: Vec<(ShaderInputType, u32)>,
        max_sets: u32,
    ) -> Result<Self, String> {
        let pool_sizes = counts
            .iter()
            .map(|(ty, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(ty.get_descriptor_type())
                    .descriptor_count(*count)
            })
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
        Ok(Self {
            painter,
            descriptor_pool,
        })
    }

    pub fn allocate(&self, layout: &ShaderInputLayout) -> Result<vk::DescriptorSet, String> {
        unsafe {
            Ok(self
                .painter
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&[layout.descriptor_set_layout]),
                )
                .map_err(|e| format!("at descriptor set allocation: {e}"))?
                .swap_remove(0))
        }
    }
}

impl Drop for ShaderInputAllocator {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
