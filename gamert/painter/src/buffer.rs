use std::sync::Arc;

use ash::vk;

use crate::Painter;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    pub painter: Arc<Painter>,
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