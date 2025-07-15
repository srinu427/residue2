use std::sync::{Arc, Mutex};

use ash::vk::{self, Handle};

use crate::{image::is_format_depth, Buffer, Image2d, ImageAccess, Painter};

pub struct Allocator {
    painter: Arc<Painter>,
    pub(crate) allocator: gpu_allocator::vulkan::Allocator,
    pub(crate) to_be_deleted: Arc<Mutex<Vec<gpu_allocator::vulkan::Allocation>>>
}

impl Allocator {
    pub fn new(painter: Arc<Painter>) -> Result<Self, String> {
        let allocator = painter.new_allocator()?;
        Ok(Self {
            painter,
            allocator,
            to_be_deleted: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn create_image_2d(
        &mut self,
        format: vk::Format,
        extent: vk::Extent2D,
        image_usage_flags: Vec<ImageAccess>,
        gpu_local: bool,
    ) -> Result<Image2d, String> {
        unsafe {
            let mut usage_flags = vk::ImageUsageFlags::empty();
            for access in image_usage_flags {
                usage_flags |= access.to_usage_flags(is_format_depth(format));
            }
            let image = self
                .painter
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
            let requirements = self.painter.device.get_image_memory_requirements(image);
            let mem_location = if gpu_local {
                gpu_allocator::MemoryLocation::GpuOnly
            } else {
                gpu_allocator::MemoryLocation::CpuToGpu
            };
            let allocation = self
                .allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:x}", image.as_raw()),
                    requirements,
                    location: mem_location,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| format!("at image allocation: {e}"))?;
            self.painter
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| format!("at image memory binding: {e}"))?;
            self.image_allocations.insert(image, allocation);
            let image_view = Image2d::create_image_view(&self.painter, image, format)
                .map_err(|e| format!("at image view creation: {e}"))?;
            let image2d = Image2d {
                image,
                painter: self.painter.clone(),
                image_view,
                format,
                extent,
                is_swapchain_image: false,
            };
            Ok(image2d)
        }
    }

    pub fn free_tbd(&mut self) -> Result<(), String> {
        let mut to_be_deleted = self.to_be_deleted.lock().map_err(|e| e.to_string())?;
        for allocation in to_be_deleted.drain(..) {
            self.allocator.free(allocation).map_err(|e| format!("at allocation free: {e}"))?;
        }
        Ok(())
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        
    }
}
