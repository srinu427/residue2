use painter::ash::{self, khr, vk};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SwapchainManagerError {
    #[error("Error querying for surface capabilities: {0}")]
    SurfaceCapsQueryError(vk::Result),
    #[error("Error creating Vulkan Swapchain: {0}")]
    CreateError(vk::Result),
    #[error("Error fetching swapchain images: {0}")]
    FetchImages(vk::Result),
    #[error("Error creating Image view for swapchain image: {0}")]
    CreateImageViewError(vk::Result),
}

pub struct SwapchainManager {
    pub image_views: Vec<vk::ImageView>,
    pub images: Vec<vk::Image>,
    pub present_mode: vk::PresentModeKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub surface: vk::SurfaceKHR,
    pub gpu: vk::PhysicalDevice,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_device: khr::swapchain::Device,
}

impl SwapchainManager {
    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            for image_view in self.image_views.drain(..) {
                device.destroy_image_view(image_view, None);
            }
            for image in self.images.drain(..) {
                device.destroy_image(image, None);
            }
            self.swapchain_device.destroy_swapchain(self.swapchain, None);
        }
    }

    pub fn add_initialize_layout_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer
    ) -> Result<(), SwapchainManagerError> {
        let image_memory_barriers = self
            .images
            .iter().
            map(|&image| {
                vk::ImageMemoryBarrier2::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            })
            .collect::<Vec<_>>();
        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_memory_barriers);
        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        Ok(())
    }

    pub fn new(
        device: &ash::Device,
        surface_instance: khr::surface::Instance,
        gpu: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        swapchain_device: khr::swapchain::Device,
        present_mode: vk::PresentModeKHR,
        surface_format: vk::SurfaceFormatKHR,
    ) -> Result<Self, SwapchainManagerError> {
        unsafe {
            let surface_caps = surface_instance
                .get_physical_device_surface_capabilities(gpu, surface)
                .map_err(SwapchainManagerError::SurfaceCapsQueryError)?;

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(surface_caps.min_image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(surface_caps.current_extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::STORAGE,
                )
                .pre_transform(surface_caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true);

            let swapchain = swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainManagerError::CreateError)?;

            let images = swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(SwapchainManagerError::FetchImages)?;

            let image_views = images
                .iter()
                .map(|&image| {
                    let image_view_create_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .format(surface_format.format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1)
                                .base_array_layer(0)
                                .base_mip_level(0),
                        );
                    device
                        .create_image_view(&image_view_create_info, None)
                        .map_err(SwapchainManagerError::CreateImageViewError)
                })
                .collect::<Result<Vec<_>, _>>()?;

                Ok(Self {
                    image_views,
                    images,
                    present_mode,
                    surface_format,
                    surface_resolution: surface_caps.current_extent,
                    surface,
                    gpu,
                    swapchain,
                    swapchain_device,
                })
        }
    }

    pub fn refresh_resolution(
        &mut self,
        device: &ash::Device,
        surface_instance: khr::surface::Instance,
        queue: vk::Queue,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), SwapchainManagerError> {
        unsafe {
            let surface_caps = surface_instance
                .get_physical_device_surface_capabilities(self.gpu, self.surface)
                .map_err(SwapchainManagerError::SurfaceCapsQueryError)?;

            let new_resolution = surface_caps.current_extent;

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(self.images.len() as u32)
                .image_format(self.surface_format.format)
                .image_color_space(self.surface_format.color_space)
                .image_extent(new_resolution)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::STORAGE,
                )
                .pre_transform(surface_caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(self.present_mode)
                .old_swapchain(self.swapchain)
                .clipped(true);

            let old_swapchain = self.swapchain;

            let new_swapchain = self
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(SwapchainManagerError::CreateError)?;

            let new_swapchain_images = self
                .swapchain_device
                .get_swapchain_images(new_swapchain)
                .map_err(SwapchainManagerError::FetchImages)?;

            let new_swapchain_image_views = new_swapchain_images
                .iter()
                .map(|&image| {
                    let image_view_create_info = vk::ImageViewCreateInfo::default()
                        .image(image)
                        .format(self.surface_format.format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1)
                                .level_count(1)
                                .base_array_layer(0)
                                .base_mip_level(0)
                        );
                    device
                        .create_image_view(&image_view_create_info, None)
                        .map_err(SwapchainManagerError::CreateImageViewError)
                })
                .collect::<Result<Vec<_>, _>>()?;

            self.swapchain = new_swapchain;
            self.images = new_swapchain_images;
            self.image_views = new_swapchain_image_views;

            self.swapchain_device.destroy_swapchain(old_swapchain, None);

            self.surface_resolution = surface_caps.current_extent;
            Ok(())
        }
    }
}
