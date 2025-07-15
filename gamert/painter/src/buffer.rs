use std::sync::{Arc, Mutex};

use ash::vk;

use crate::{Allocator, Painter};

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    bound_mem: Option<(Arc<Mutex<Vec<gpu_allocator::vulkan::Allocation>>>, gpu_allocator::vulkan::Allocation)>,
    pub painter: Arc<Painter>,
}

impl Buffer {
    pub fn new(
        painter: Arc<Painter>,
        size: u64,
        buffer_usage_flags: vk::BufferUsageFlags,
    ) -> Result<Self, String> {

        let buffer = unsafe {
            painter
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .usage(buffer_usage_flags)
                        .size(size),
                    None,
                )
                .map_err(|e| format!("at buffer creation: {e}"))?
        };
        Ok(Self {
            buffer,
            size,
            bound_mem: None,
            painter,
        })
    
    }

    pub fn bna_memory(&mut self, allocator: &mut Allocator) -> Result<(), String> {
        if self.bound_mem.is_some() {
            return Ok(());
        }

        let requirements = unsafe { self.painter.device.get_buffer_memory_requirements(self.buffer) };
        let allocation = allocator
            .allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: &format!("{:?}", self.buffer),
                requirements,
                location: gpu_allocator::MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| format!("at buffer allocation: {e}"))?;

        unsafe {
            self.painter
                .device
                .bind_buffer_memory(self.buffer, allocation.memory(), allocation.offset())
                .map_err(|e| format!("at buffer memory binding: {e}"))?;
        }

        self.bound_mem = Some((allocator.to_be_deleted.clone(), allocation));
        Ok(())
    }

    pub fn write_data(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() as u64 > self.size {
            return Err("Data size exceeds buffer size".to_string());
        }
        let allocation = &mut self.bound_mem.as_mut()
            .ok_or("Buffer memory is not bound")?
            .1;

        let data_ptr = allocation
            .mapped_slice_mut()
            .ok_or("buffer memory cant be mapped as mutable")?;
        data_ptr[..data.len()].copy_from_slice(data);

        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .destroy_buffer(self.buffer, None);
        }
    }
}