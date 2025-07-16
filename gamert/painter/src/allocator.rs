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

    pub fn free_tbd(&mut self) -> Result<(), String> {
        let mut to_be_deleted = self.to_be_deleted.lock().map_err(|e| e.to_string())?;
        for allocation in to_be_deleted.drain(..) {
            self.allocator.free(allocation).map_err(|e| format!("at allocation free: {e}"))?;
        }
        Ok(())
    }
}
