use std::sync::Arc;

use ash::vk;

use crate::Painter;

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
            ImageAccess::PipelineAttachment => if is_depth_format {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            },
            ImageAccess::Present => vk::AccessFlags::MEMORY_READ,
        }
    }

    pub fn to_usage_flags(&self, is_depth_format: bool) -> vk::ImageUsageFlags {
        match self {
            ImageAccess::None => vk::ImageUsageFlags::empty(),
            ImageAccess::TransferRead => vk::ImageUsageFlags::TRANSFER_SRC,
            ImageAccess::TransferWrite => vk::ImageUsageFlags::TRANSFER_DST,
            ImageAccess::ShaderRead => vk::ImageUsageFlags::SAMPLED,
            ImageAccess::PipelineAttachment => if is_depth_format {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE
            },
            ImageAccess::Present => vk::ImageUsageFlags::empty(),
        }
    }

    pub fn get_image_layout(&self, is_depth_format: bool) -> vk::ImageLayout {
        match self {
            ImageAccess::None => vk::ImageLayout::UNDEFINED,
            ImageAccess::TransferRead => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageAccess::TransferWrite => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageAccess::ShaderRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageAccess::PipelineAttachment => if is_depth_format {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            },
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

pub struct Image2d{
    pub image_views: Vec<vk::ImageView>,
    pub image: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub is_swapchain_image: bool,
    pub painter: Arc<Painter>,
}

impl Image2d {
    pub fn create_image_view(&mut self) -> Result<vk::ImageView, String> {
        unsafe {
            let image_view = self
                .painter
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(self.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(self.format)
                        .components(vk::ComponentMapping::default())
                        .subresource_range(self.get_subresource_range()),
                    None,
                )
                .map_err(|e| format!("at image view creation: {e}"))?;
            self.image_views.push(image_view);
            Ok(image_view)
        }
    }

    pub fn get_subresource(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(get_image_aspect(self.format))
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1)
    }

    pub fn get_subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(get_image_aspect(self.format))
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
    }

    pub fn get_full_size_offset(&self) -> [vk::Offset3D; 2] {
        [
            vk::Offset3D {x: 0, y: 0, z: 0},
            vk::Offset3D {x: self.extent.width as _, y: self.extent.height as _, z: 1},
        ]
    }

    pub fn extent3d(&self) -> vk::Extent3D {
        vk::Extent3D{width: self.extent.width, height: self.extent.height, depth: 1}
    }
}

impl Drop for Image2d {
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.image_views {
                self.painter.device.destroy_image_view(*image_view, None);
            }
            if !self.is_swapchain_image {
                self.painter.device.destroy_image(self.image, None);
            }
        }
    }
}