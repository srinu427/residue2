use std::sync::Arc;

use ash::vk;

use crate::Painter;

pub struct CpuFuture {
    pub fence: vk::Fence,
    painter: Arc<Painter>,
}

impl CpuFuture {
    pub fn new(painter: Arc<Painter>, signaled: bool) -> Result<Self, String> {
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
            Ok(Self { fence, painter })
        }
    }

    pub fn wait(&self) -> Result<(), String> {
        unsafe {
            self.painter
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("at fence wait: {e}"))?;
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<(), String> {
        unsafe {
            self.painter
                .device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("at fence reset: {e}"))?;
        }
        Ok(())
    }

    pub fn wait_and_reset(&self) -> Result<(), String> {
        self.wait()?;
        self.reset()?;
        Ok(())
    }
}

impl Drop for CpuFuture {
    fn drop(&mut self) {
        unsafe {
            self.painter.device.destroy_fence(self.fence, None);
        }
    }
}

pub struct GpuFuture {
    pub semaphore: vk::Semaphore,
    painter: Arc<Painter>,
}

impl GpuFuture {
    pub fn new(painter: Arc<Painter>) -> Result<Self, String> {
        unsafe {
            let semaphore = painter
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|e| format!("at semaphore creation: {e}"))?;
            Ok(Self { semaphore, painter })
        }
    }
}

impl Drop for GpuFuture {
    fn drop(&mut self) {
        unsafe {
            self.painter.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
