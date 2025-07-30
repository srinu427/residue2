use ash::vk;
use thiserror::Error;

use crate::renderers::mesh_renderer::pipeline_info::PipelineInfo;

#[derive(Error, Debug)]
pub enum OutputInfoCreateError {
    #[error("Failed to create image ({0}): {1}")]
    ImageCreationFailed(&'static str, vk::Result),
    #[error("Failed to allocate image memory ({0}): {1}")]
    ImageMemAllocationFailed(&'static str, gpu_allocator::AllocationError),
    #[error("Failed to allocate image memory ({0}): {1}")]
    ImageViewCreationFailed(&'static str, vk::Result),
    #[error("Failed to create framebuffer: {0}")]
    FramebufferCreationFailed(vk::Result),
}

pub struct OutputInfo {
    resolution: vk::Extent2D,
    color_image: vk::Image,
    color_image_view: vk::ImageView,
    color_image_allocation: Option<gpu_allocator::vulkan::Allocation>,
    depth_image: vk::Image,
    depth_image_view: vk::ImageView,
    depth_image_allocation: Option<gpu_allocator::vulkan::Allocation>,
    framebuffer: vk::Framebuffer,
}

impl OutputInfo {
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        unsafe {
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.color_image_view, None);
            device.destroy_image(self.color_image, None);
            device.destroy_image_view(self.depth_image_view, None);
            device.destroy_image(self.depth_image, None);
        }
        if let Some(allocation) = self.color_image_allocation.take() {
            allocator.free(allocation);
        }
        if let Some(allocation) = self.depth_image_allocation.take() {
            allocator.free(allocation);
        }
    }
}

impl PipelineInfo {
    pub fn create_output_info(
        &self,
        resolution: vk::Extent2D,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator
    ) -> Result<OutputInfo, OutputInfoCreateError> {
        // Color image and view creation
        let color_image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: resolution.width,
                height: resolution.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);

        let color_image = unsafe {
            device
                .create_image(&color_image_create_info, None)
                .map_err(|e| OutputInfoCreateError::ImageCreationFailed("Color", e))?
        };

        let color_image_allocation = unsafe {
            allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:?}", color_image),
                    requirements: device.get_image_memory_requirements(color_image),
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| OutputInfoCreateError::ImageMemAllocationFailed("Color", e))?
        };

        let color_image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(color_image)
            .format(vk::Format::R8G8B8A8_UNORM)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1)
                    .base_array_layer(0)
                    .base_mip_level(0),
            );
        
        let color_image_view = unsafe {
            device
                .create_image_view(&color_image_view_create_info, None)
                .map_err(|e| OutputInfoCreateError::ImageViewCreationFailed("Color", e))?
        };

        // Depth image and view creation
        let depth_image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(self.depth_format)
            .extent(vk::Extent3D {
                width: resolution.width,
                height: resolution.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC);

        let depth_image = unsafe {
            device
                .create_image(&depth_image_create_info, None)
                .map_err(|e| OutputInfoCreateError::ImageCreationFailed("Depth", e))?
        };

        let depth_image_allocation = unsafe {
            allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:?}", depth_image),
                    requirements: device.get_image_memory_requirements(depth_image),
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| OutputInfoCreateError::ImageMemAllocationFailed("Depth", e))?
        };

        let depth_image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(depth_image)
            .format(self.depth_format)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    .layer_count(1)
                    .level_count(1)
                    .base_array_layer(0)
                    .base_mip_level(0),
            );
        
        let depth_image_view = unsafe {
            device
                .create_image_view(&depth_image_view_create_info, None)
                .map_err(|e| OutputInfoCreateError::ImageViewCreationFailed("Depth", e))?
        };

        // Framebuffer creation
        let framebuffer = unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&[color_image_view, depth_image_view])
                        .width(resolution.width)
                        .height(resolution.height)
                        .layers(1),
                    None
                )
                .map_err(OutputInfoCreateError::FramebufferCreationFailed)?
        };

        Ok(OutputInfo {
            resolution,
            color_image,
            color_image_view,
            color_image_allocation: Some(color_image_allocation),
            depth_image,
            depth_image_view,
            depth_image_allocation: Some(depth_image_allocation),
            framebuffer,
        })
    }
}