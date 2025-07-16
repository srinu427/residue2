use ash::vk;
use crossbeam::channel::Sender;
use thiserror::Error;

use crate::{
    GAllocator, Painter,
    allocator::{GAllocatorError, RawAllocation},
    painter::PainterDelete,
};

#[derive(Debug, Error)]
pub enum BufferError {
    #[error("Error creating Vulkan buffer: {0}")]
    CreateError(vk::Result),
    #[error("Error allocating memory: {0}")]
    MemoryAllocationError(GAllocatorError),
    #[error("No memory allocated to this buffer")]
    MemoryNotAllocatedError,
    #[error("Error binding allocated memory to buffer: {0}")]
    MemoryBindError(vk::Result),
    #[error("Buffer is not host visible/writable")]
    MemoryWriteError,
}

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    bound_mem: Option<RawAllocation>,
    delete_sender: Sender<PainterDelete>,
}

impl Buffer {
    pub fn write_to_mem(&mut self, data: &[u8]) -> Result<(), BufferError> {
        let mapped_ptr = self
            .bound_mem
            .as_mut()
            .ok_or(BufferError::MemoryNotAllocatedError)?
            .mapped_slice_mut()
            .ok_or(BufferError::MemoryWriteError)?;
        mapped_ptr[..data.len()].copy_from_slice(data);
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self
            .delete_sender
            .try_send(PainterDelete::Buffer(self.buffer))
            .inspect_err(|e| {
                eprintln!(
                    "error sending drop signal for buffer {:?}: {e}",
                    self.buffer
                )
            });
    }
}

impl Painter {
    pub fn new_buffer(
        &self,
        size: u64,
        buffer_usage_flags: vk::BufferUsageFlags,
        mem_allocator: Option<&mut GAllocator>,
        mem_host_visible: Option<bool>,
    ) -> Result<Buffer, BufferError> {
        let buffer = unsafe {
            self.device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .usage(buffer_usage_flags)
                        .size(size),
                    None,
                )
                .map_err(BufferError::CreateError)?
        };

        if let Some(mem_allocator) = mem_allocator {
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let gpu_local = !mem_host_visible.unwrap_or(false);
            let allocation = mem_allocator
                .allocate_mem(&format!("{:?}", buffer), requirements, gpu_local)
                .map_err(BufferError::MemoryAllocationError)?;
            unsafe {
                self.device
                    .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                    .map_err(BufferError::MemoryBindError)?;
            }
        }

        Ok(Buffer {
            buffer,
            size,
            bound_mem: None,
            delete_sender: self.delete_signal_sender.clone(),
        })
    }
}
