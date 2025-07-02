use std::sync::Arc;

use ash::vk::{self, Handle};
use hashbrown::HashMap;

use crate::{image::is_format_depth, Image2d, ImageAccess, Painter};

pub struct Allocator {
    painter: Arc<Painter>,
    allocator: gpu_allocator::vulkan::Allocator,
    buffer_allocations: HashMap<vk::Buffer, gpu_allocator::vulkan::Allocation>,
    image_allocations: HashMap<vk::Image, gpu_allocator::vulkan::Allocation>,
}

impl Allocator {
    pub fn new(painter: Arc<Painter>) -> Result<Self, String> {
        let allocator = painter.new_allocator()?;
        Ok(Self {
            painter,
            allocator,
            buffer_allocations: HashMap::new(),
            image_allocations: HashMap::new(),
        })
    }

    pub fn create_buffer(
        &mut self,
        size: u64,
        buffer_usage_flags: vk::BufferUsageFlags,
        gpu_local: bool,
    ) -> Result<vk::Buffer, String> {
        unsafe {
            let buffer = self
                .painter
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .usage(buffer_usage_flags)
                        .size(size),
                    None,
                )
                .map_err(|e| format!("at buffer creation: {e}"))?;
            let requirements = self.painter.device.get_buffer_memory_requirements(buffer);
            let mem_location = if gpu_local {
                gpu_allocator::MemoryLocation::GpuOnly
            } else {
                gpu_allocator::MemoryLocation::CpuToGpu
            };
            let allocation = self
                .allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:x}", buffer.as_raw()),
                    requirements,
                    location: mem_location,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| format!("at buffer allocation: {e}"))?;

            self.painter
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| format!("at buffer memory binding: {e}"))?;

            self.buffer_allocations.insert(buffer, allocation);
            Ok(buffer)
        }
    }

    pub fn write_to_buffer_mem(&mut self, buffer: vk::Buffer, data: &[u8]) -> Result<(), String> {
        let allocation = self
            .buffer_allocations
            .get_mut(&buffer)
            .ok_or("buffer not found")?;
        let data_ptr = allocation
            .mapped_slice_mut()
            .ok_or("buffer memory cant be mapped as mutable")?;
        data_ptr[..data.len()].copy_from_slice(data);
        Ok(())
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
            let image2d = Image2d { image, painter: self.painter.clone(), image_views: vec![], format, extent, is_swapchain_image: false };
            Ok(image2d)
        }
    }

    fn delete_allocations_inner(
        &mut self,
        buffers: &[vk::Buffer],
        images: &[vk::Image],
    ) -> Result<(), String> {
        for buffer in buffers {
            if let Some(allocation) = self.buffer_allocations.remove(buffer) {
                self.allocator
                    .free(allocation)
                    .map_err(|e| format!("at buffer deallocation: {e}"))?;
            }
        }
        for image in images {
            if let Some(allocation) = self.image_allocations.remove(image) {
                self.allocator
                    .free(allocation)
                    .map_err(|e| format!("at image deallocation: {e}"))?;
            }
        }
        Ok(())
    }

    pub fn delete_allocations(
        &mut self,
        buffers: &[vk::Buffer],
        images: &[&Image2d],
    ) -> Result<(), String> {
        let images = images.iter().map(|image| image.image).collect::<Vec<_>>();
        self.delete_allocations_inner(buffers, &images)
            .inspect_err(|e| eprintln!("at buffer and image deallocation: {e}"))
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        let _ = self
            .delete_allocations_inner(
                &self.buffer_allocations.keys().cloned().collect::<Vec<_>>(),
                &self.image_allocations.keys().cloned().collect::<Vec<_>>(),
            )
            .inspect_err(|e| eprintln!("at buffer and image deallocation: {e}"));
    }
}
