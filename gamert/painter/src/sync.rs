use ash::vk;
use crossbeam::channel::Sender;
use thiserror::Error;

use crate::{painter::PainterDelete, Painter};

#[derive(Debug, Error)]
pub enum CpuFutureError {
    #[error("Error creating Vulkan Fence")]
    CreateError(vk::Result),
    #[error("Error waiting for Vulkan Fence")]
    WaitError(vk::Result),
    #[error("Error resetting Vulkan Fence")]
    ResetError(vk::Result)
}

pub struct CpuFuture {
    pub fence: vk::Fence,
    delete_sender: Sender<PainterDelete>,
}

impl CpuFuture {
    pub fn new(painter: &Painter, signaled: bool) -> Result<Self, String> {
        unsafe {
            let create_flags = if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            };
            let fence = painter
                .device
                .create_fence(&vk::FenceCreateInfo::default().flags(create_flags), None)
                .map_err(|e| format!("at fence creation: {e}"))?;
            Ok(Self { fence, delete_sender: painter.delete_signal_sender.clone() })
        }
    }

    
}

impl Drop for CpuFuture {
    fn drop(&mut self) {
        let _ = self
            .delete_sender
            .try_send(PainterDelete::Fence(self.fence))
            .inspect_err(|e| {
                eprintln!(
                    "error sending drop signal for fence {:?}: {e}",
                    self.fence
                )
            });
    }
}

impl Painter {
    pub fn cpu_future_wait(&self, cpu_future: &CpuFuture) -> Result<(), CpuFutureError> {
        unsafe {
            self
                .device
                .wait_for_fences(&[cpu_future.fence], true, u64::MAX)
                .map_err(CpuFutureError::CreateError)?;
        }
        Ok(())
    }

    pub fn cpu_future_reset(&self, cpu_future: &CpuFuture) -> Result<(), CpuFutureError> {
        unsafe {
            self
                .device
                .reset_fences(&[cpu_future.fence])
                .map_err(CpuFutureError::ResetError)?;
        }
        Ok(())
    }

    pub fn wait_and_reset(&self, cpu_future: &CpuFuture) -> Result<(), CpuFutureError> {
        self.cpu_future_wait(cpu_future)?;
        self.cpu_future_reset(cpu_future)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum GpuFutureError {
    #[error("Error creating Vulkan Semaphore")]
    CreateError(vk::Result)
}

pub struct GpuFuture {
    pub semaphore: vk::Semaphore,
    delete_sender: Sender<PainterDelete>,
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        let _ = self
            .delete_sender
            .try_send(PainterDelete::Semaphore(self.semaphore))
            .inspect_err(|e| {
                eprintln!(
                    "error sending drop signal for semaphore {:?}: {e}",
                    self.semaphore
                )
            });
    }
}

impl Painter {
    pub fn new_gpu_future(&self) -> Result<GpuFuture, GpuFutureError> {
        let semaphore = unsafe {
            self
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(GpuFutureError::CreateError)?
        };
        Ok(GpuFuture { semaphore, delete_sender: self.delete_signal_sender.clone() })
    }
}
