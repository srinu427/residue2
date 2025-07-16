use std::sync::Arc;

use ash::vk;
use crossbeam::channel::Receiver;
use crossbeam::channel::Sender;
use thiserror::Error;

use crate::Painter;

pub use gpu_allocator::vulkan::Allocation as RawAllocation;
pub use gpu_allocator::vulkan::Allocator as RawAllocator;

#[derive(Debug, Error)]
pub enum GAllocatorError {
    #[error("Error creating GPU Memory Allocator: {0}")]
    CreateError(gpu_allocator::AllocationError),
    #[error("Error at allocating memory: {0}")]
    MemoryAllocationError(gpu_allocator::AllocationError),
    #[error("Error freeing GPU Memory: {0}")]
    MemoryFreeError(gpu_allocator::AllocationError),
    #[error("Specified Allocator ID is not found")]
    MemoryNotFound,
    #[error("Memory is not host visible")]
    MemoryNotWritable,
    #[error("Error writing to memeory")]
    MemoryWriteError,
    #[error("Error getting lock to to-be-deleted list for the allocator: {0}")]
    TbdListLockError(String),
}

pub struct GAllocator {
    painter: Arc<Painter>,
    pub(crate) allocator: RawAllocator,
    delete_event_receiver: Receiver<RawAllocation>,
    delete_event_sender: Sender<RawAllocation>,
}

impl GAllocator {
    pub fn new(painter: Arc<Painter>) -> Result<Self, GAllocatorError> {
        let allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: painter.instance.clone(),
                device: painter.device.clone(),
                physical_device: painter.physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(GAllocatorError::CreateError)?;
        let (s, r) = crossbeam::channel::unbounded();
        Ok(Self {
            painter,
            allocator,
            delete_event_receiver: r,
            delete_event_sender: s,
        })
    }

    pub fn process_free_events(&mut self) -> Result<(), GAllocatorError> {
        loop {
            let Ok(tbd) = self.delete_event_receiver.try_recv() else {
                break;
            };
            self.allocator
                .free(tbd)
                .map_err(GAllocatorError::MemoryFreeError)?;
        }
        Ok(())
    }

    pub fn allocate_mem(
        &mut self,
        name: &str,
        requirements: vk::MemoryRequirements,
        gpu_local: bool,
    ) -> Result<RawAllocation, GAllocatorError> {
        let location = if gpu_local {
            gpu_allocator::MemoryLocation::GpuOnly
        } else {
            gpu_allocator::MemoryLocation::CpuToGpu
        };
        let allocation = self
            .allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(GAllocatorError::MemoryAllocationError)?;
        Ok(allocation)
    }
}
