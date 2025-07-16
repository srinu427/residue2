use ash::vk;
use crossbeam::channel::Sender;
use thiserror::Error;

use crate::{
    GAllocator, Painter,
    allocator::{GAllocatorError, RawAllocation},
    painter::PainterDelete,
};

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

#[derive(Debug, Error)]
pub enum Image2dError {
    #[error("Error creating Vulkan 2D Image: {0}")]
    CreateError(vk::Result),
    #[error("Error creating a View for the Image created: {0}")]
    ViewCreateError(vk::Result),
    #[error("Error from memory allocator: {0}")]
    MemoryAllocationError(GAllocatorError),
    #[error("Allocated memory not found with allocator")]
    MemoryNotFoundError,
    #[error("Error binding allocated memory to image: {0}")]
    MemoryBindError(vk::Result),
}

pub struct Image2d {
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub(crate) bound_mem: Option<RawAllocation>,
    pub(crate) delete_sender: Option<Sender<PainterDelete>>,
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
    ) -> Result<vk::ImageView, Image2dError> {
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
                .map_err(Image2dError::ViewCreateError)
        }
    }
}

impl Drop for Image2d {
    fn drop(&mut self) {
        let Some(delete_sender) = self.delete_sender.take() else {
            return;
        };
        let _ = delete_sender
            .try_send(PainterDelete::ImageView(self.image_view))
            .inspect_err(|e| {
                eprintln!(
                    "error sending drop signal for image view {:?}: {e}",
                    self.image_view
                )
            });
        let _ = delete_sender
            .try_send(PainterDelete::Image(self.image))
            .inspect_err(|e| {
                eprintln!("error sending drop signal for image {:?}: {e}", self.image)
            });
    }
}

impl Painter {
    pub fn new_image_2d(
        &self,
        format: vk::Format,
        extent: vk::Extent2D,
        image_usage_flags: Vec<ImageAccess>,
        mem_allocator: Option<&mut GAllocator>,
        mem_host_visible: Option<bool>,
    ) -> Result<Image2d, Image2dError> {
        let mut usage_flags = vk::ImageUsageFlags::empty();
        for access in image_usage_flags {
            usage_flags |= access.to_usage_flags(is_format_depth(format));
        }
        let image = unsafe {
            self.device
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
                .map_err(Image2dError::CreateError)?
        };

        let image_view = unsafe {
            self.device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(Image2d::make_subresource_range(format)),
                    None,
                )
                .map_err(Image2dError::ViewCreateError)?
        };

        let bound_mem = match mem_allocator {
            Some(mem_allocator) => {
                let requirements = unsafe { self.device.get_image_memory_requirements(image) };
                let gpu_local = !mem_host_visible.unwrap_or(false);
                let allocation = mem_allocator
                    .allocate_mem(&format!("{:?}", image), requirements, gpu_local)
                    .map_err(Image2dError::MemoryAllocationError)?;
                unsafe {
                    self.device
                        .bind_image_memory(image, allocation.memory(), allocation.offset())
                        .map_err(Image2dError::MemoryBindError)?;
                }
                Some(allocation)
            }
            None => None,
        };
        Ok(Image2d {
            image_view,
            image,
            format,
            extent,
            bound_mem,
            delete_sender: Some(self.delete_signal_sender.clone()),
        })
    }
}
