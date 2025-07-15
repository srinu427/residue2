use std::sync::{Arc, Mutex};

use ash::vk;

use crate::{Allocator, Painter};

pub fn is_format_depth(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM
            | vk::Format::D32_SFLOAT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn has_stencil_component(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn get_image_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if is_format_depth(format) {
        if has_stencil_component(format) {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            vk::ImageAspectFlags::DEPTH
        }
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageAccess {
    None,
    TransferRead,
    TransferWrite,
    ShaderRead,
    PipelineAttachment,
    Present,
}

impl ImageAccess {
    pub fn to_access_flags(&self, is_depth_format: bool) -> vk::AccessFlags {
        match self {
            ImageAccess::None => vk::AccessFlags::empty(),
            ImageAccess::TransferRead => vk::AccessFlags::TRANSFER_READ,
            ImageAccess::TransferWrite => vk::AccessFlags::TRANSFER_WRITE,
            ImageAccess::ShaderRead => vk::AccessFlags::SHADER_READ,
            ImageAccess::PipelineAttachment => {
                if is_depth_format {
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                } else {
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                }
            }
            ImageAccess::Present => vk::AccessFlags::MEMORY_READ,
        }
    }

    pub fn to_usage_flags(&self, is_depth_format: bool) -> vk::ImageUsageFlags {
        match self {
            ImageAccess::None => vk::ImageUsageFlags::empty(),
            ImageAccess::TransferRead => vk::ImageUsageFlags::TRANSFER_SRC,
            ImageAccess::TransferWrite => vk::ImageUsageFlags::TRANSFER_DST,
            ImageAccess::ShaderRead => vk::ImageUsageFlags::SAMPLED,
            ImageAccess::PipelineAttachment => {
                if is_depth_format {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE
                }
            }
            ImageAccess::Present => vk::ImageUsageFlags::empty(),
        }
    }

    pub fn get_image_layout(&self, is_depth_format: bool) -> vk::ImageLayout {
        match self {
            ImageAccess::None => vk::ImageLayout::UNDEFINED,
            ImageAccess::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageAccess::TransferWrite => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageAccess::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageAccess::PipelineAttachment => {
                if is_depth_format {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                }
            }
            ImageAccess::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    pub fn get_pipeline_stage(&self) -> vk::PipelineStageFlags {
        match self {
            ImageAccess::None => vk::PipelineStageFlags::TOP_OF_PIPE,
            ImageAccess::TransferRead => vk::PipelineStageFlags::TRANSFER,
            ImageAccess::TransferWrite => vk::PipelineStageFlags::TRANSFER,
            ImageAccess::ShaderRead => vk::PipelineStageFlags::FRAGMENT_SHADER,
            ImageAccess::PipelineAttachment => vk::PipelineStageFlags::ALL_GRAPHICS,
            ImageAccess::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        }
    }
}

pub struct Image2d {
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub is_swapchain_image: bool,
    pub(crate) bound_mem: Option<(Arc<Mutex<Vec<gpu_allocator::vulkan::Allocation>>>, gpu_allocator::vulkan::Allocation)>,
    pub painter: Arc<Painter>,
}

impl Image2d {
    pub(crate) fn make_subresource(format: vk::Format) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(get_image_aspect(format))
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1)
    }

    pub fn get_subresource_range(&self) -> vk::ImageSubresourceRange {
        Self::make_subresource_range(self.format)
    }

    pub(crate) fn make_subresource_range(format: vk::Format) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(get_image_aspect(format))
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
    }

    pub fn get_subresource_layers(&self) -> vk::ImageSubresourceLayers {
        Self::make_subresource(self.format)
    }

    pub fn get_full_size_offset(&self) -> [vk::Offset3D; 2] {
        [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: self.extent.width as _,
                y: self.extent.height as _,
                z: 1,
            },
        ]
    }

    pub fn extent3d(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.extent.width,
            height: self.extent.height,
            depth: 1,
        }
    }

    pub fn create_image_view(
        painter: &Painter,
        image: vk::Image,
        format: vk::Format,
    ) -> Result<vk::ImageView, String> {
        unsafe {
            painter
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(Self::make_subresource_range(format)),
                    None,
                )
                .map_err(|e| format!("at image view creation: {e}"))
        }
    }

    pub fn new(painter: Arc<Painter>,
        format: vk::Format,
        extent: vk::Extent2D,
        image_usage_flags: Vec<ImageAccess>
    ) -> Result<Self, String> {
        unsafe {
            let mut usage_flags = vk::ImageUsageFlags::empty();
            for access in image_usage_flags {
                usage_flags |= access.to_usage_flags(is_format_depth(format));
            }
            let image = painter
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .format(format)
                        .extent(vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .usage(usage_flags)
                        .image_type(vk::ImageType::TYPE_2D)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    None,
                )
                .map_err(|e| format!("at image creation: {e}"))?;
            let image_view = Self::create_image_view(&painter, image, format)
                .map_err(|e| format!("at image view creation: {e}"))?;
            Ok(Self { image_view, image, format, extent, is_swapchain_image: false, bound_mem: None, painter })
        }
    }

    pub fn bna_memory(&mut self, allocator: &mut Allocator) -> Result<(), String> {
        if self.bound_mem.is_some() {
            return Ok(());
        }

        let requirements = unsafe { self.painter.device.get_image_memory_requirements(self.image) };
        let allocation = allocator
            .allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: &format!("{:?}", self.image),
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| format!("at image allocation: {e}"))?;

        unsafe {
            self.painter
                .device
                .bind_image_memory(self.image, allocation.memory(), allocation.offset())
                .map_err(|e| format!("at image bind memory: {e}"))?;
        }

        self.bound_mem = Some((allocator.to_be_deleted.clone(), allocation));
        Ok(())
    }

    pub fn unbind_memory(&mut self) -> Result<(), String> {
        let Some((to_be_deleted, allocation)) = self.bound_mem.take() else { return Ok(()); };
        to_be_deleted.lock().map_err(|e| format!("at locking memorly free list: {e}"))?.push(allocation);
        Ok(())
    }
}

impl Drop for Image2d {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .destroy_image_view(self.image_view, None);

            if !self.is_swapchain_image {
                self.painter.device.destroy_image(self.image, None);
            }
        }
    }
}
