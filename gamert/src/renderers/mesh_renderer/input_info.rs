use ash::vk;
use thiserror::Error;

use crate::{renderers::mesh_renderer::Renderer, scene_elements::camera::Camera};

#[repr(C)]
pub struct ObjectInfo {
    sampler_id: u32,
    texture_id: u32,
}

#[repr(C)]
pub struct GpuVertex {
    pos: [f32; 3],
    tex_coords: [f32; 2],
    obj_id: u32,
}

#[derive(Error, Debug)]
pub enum PerFrameInputInfoCreateError {
    #[error("Failed to create buffer ({0}): {1}")]
    BufferCreationFailed(&'static str, vk::Result),
    #[error("Failed to allocate buffer memory ({0}): {1}")]
    MemoryAllocationFailed(&'static str, gpu_allocator::AllocationError),
    #[error("Failed to bind buffer memory ({0}): {1}")]
    MemoryBindingFailed(&'static str, vk::Result),
}

pub struct PerFrameInputInfo {
    max_vertices: u32,
    max_objects: u32,
    vertex_buffer: vk::Buffer,
    vertex_buffer_allocation: Option<gpu_allocator::vulkan::Allocation>,
    camera_buffer: vk::Buffer,
    camera_buffer_allocation: Option<gpu_allocator::vulkan::Allocation>,
    object_buffer: vk::Buffer,
    object_buffer_allocation: Option<gpu_allocator::vulkan::Allocation>,
}

impl PerFrameInputInfo {
    pub fn cleanup(&mut self, device: &ash::Device, allocator: &mut gpu_allocator::vulkan::Allocator) {
        unsafe {
            device.destroy_buffer(self.vertex_buffer, None);
            device.destroy_buffer(self.camera_buffer, None);
            device.destroy_buffer(self.object_buffer, None);
        }
        if let Some(allocation) = self.vertex_buffer_allocation.take() {
            allocator.free(allocation);
        }
        if let Some(allocation) = self.camera_buffer_allocation.take() {
            allocator.free(allocation);
        }
        if let Some(allocation) = self.object_buffer_allocation.take() {
            allocator.free(allocation);
        }
    }
}

impl Renderer {
    pub fn create_per_frame_input_info(
        &self,
        max_vertices: u32,
        max_objects: u32,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> Result<PerFrameInputInfo, PerFrameInputInfoCreateError> {
        let vertex_buffer_create_info = vk::BufferCreateInfo::default()
            .size(max_vertices as u64 * std::mem::size_of::<GpuVertex>() as u64)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vertex_buffer = unsafe {
            device
                .create_buffer(&vertex_buffer_create_info, None)
                .map_err(|e| PerFrameInputInfoCreateError::BufferCreationFailed("Vertex", e))?
        };

        let camera_buffer_create_info = vk::BufferCreateInfo::default()
            .size(std::mem::size_of::<Camera>() as u64)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let camera_buffer = unsafe {
            device
                .create_buffer(&camera_buffer_create_info, None)
                .map_err(|e| PerFrameInputInfoCreateError::BufferCreationFailed("Camera", e))?
        };

        let object_buffer_create_info = vk::BufferCreateInfo::default()
            .size(max_objects as u64 * std::mem::size_of::<ObjectInfo>() as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let object_buffer = unsafe {
            device
                .create_buffer(&object_buffer_create_info, None)
                .map_err(|e| PerFrameInputInfoCreateError::BufferCreationFailed("Object", e))?
        };

        unsafe {
            let vertex_buffer_allocation = allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:?}", vertex_buffer),
                    requirements: device.get_buffer_memory_requirements(vertex_buffer),
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| PerFrameInputInfoCreateError::MemoryAllocationFailed("Vertex", e))?;

            let camera_buffer_allocation = allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:?}", camera_buffer),
                    requirements: device.get_buffer_memory_requirements(camera_buffer),
                    location: gpu_allocator::MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| PerFrameInputInfoCreateError::MemoryAllocationFailed("Camera", e))?;

            let object_buffer_allocation = allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: &format!("{:?}", object_buffer),
                    requirements: device.get_buffer_memory_requirements(object_buffer),
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| PerFrameInputInfoCreateError::MemoryAllocationFailed("Object", e))?;

            device.bind_buffer_memory(
                vertex_buffer,
                vertex_buffer_allocation.memory(),
                vertex_buffer_allocation.offset(),
            )
                .map_err(|e| PerFrameInputInfoCreateError::MemoryBindingFailed("Vertex", e))?;
            device.bind_buffer_memory(
                camera_buffer,
                camera_buffer_allocation.memory(),
                camera_buffer_allocation.offset(),
            )
                .map_err(|e| PerFrameInputInfoCreateError::MemoryBindingFailed("Camera", e))?;
            device.bind_buffer_memory(
                object_buffer,
                object_buffer_allocation.memory(),
                object_buffer_allocation.offset(),
            )
                .map_err(|e| PerFrameInputInfoCreateError::MemoryBindingFailed("Object", e))?;

            Ok(PerFrameInputInfo {
                max_vertices,
                max_objects,
                vertex_buffer,
                vertex_buffer_allocation: Some(vertex_buffer_allocation),
                camera_buffer,
                camera_buffer_allocation: Some(camera_buffer_allocation),
                object_buffer,
                object_buffer_allocation: Some(object_buffer_allocation),
            })
        }
    }
}
