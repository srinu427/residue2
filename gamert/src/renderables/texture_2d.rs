use painter::gpu_allocator::vulkan::{Allocation, Allocator};
use painter::ash::{self, vk};

pub struct Texture2D {
    image: vk::Image,
    image_view: vk::ImageView,
    allocation: Option<Allocation>
}

impl Texture2D {
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.image_view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            allocator.free(allocation);
        }
    }
}