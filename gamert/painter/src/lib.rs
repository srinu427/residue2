use std::sync::Arc;

use ash::vk;

pub use ash;
pub use gpu_allocator;
pub use slotmap;
pub use winit;

mod allocator;
mod command;
mod image;
mod painter;
mod render_pipeline;
mod shader_input;
mod sheets;
mod sync;

pub use allocator::Allocator;
pub use command::{CommandBuffer, CommandPool, GpuCommand, GpuRenderPassCommand};
pub use image::{Image2d, ImageAccess};
pub use painter::Painter;
pub use render_pipeline::{RenderOutput, SingePassRenderPipeline};
pub use shader_input::{
    ShaderInputAllocator, ShaderInputBindingInfo, ShaderInputLayout, ShaderInputType,
};
pub use sheets::Sheets;
pub use sync::{CpuFuture, GpuFuture};

pub struct ShaderModule {
    pub shader_module: vk::ShaderModule,
    painter: Arc<Painter>,
}

impl ShaderModule {
    pub fn new(painter: Arc<Painter>, code: &[u8]) -> Result<Self, String> {
        unsafe {
            let shader_module = painter
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(code.align_to::<u32>().1),
                    None,
                )
                .map_err(|e| format!("at shader module creation: {e}"))?;
            Ok(Self {
                shader_module,
                painter: painter.clone(),
            })
        }
    }

    pub fn get_vk(&self) -> &vk::ShaderModule {
        &self.shader_module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}
