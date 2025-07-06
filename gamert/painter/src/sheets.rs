use std::sync::Arc;

use ash::{khr, vk};

use crate::{CommandBuffer, CpuFuture, GpuCommand, GpuFuture, Image2d, ImageAccess, Painter};

pub struct Sheets {
    pub swapchain_images: Vec<Image2d>,
    pub present_mode: vk::PresentModeKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_resolution: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_device: khr::swapchain::Device,
    pub painter: Arc<Painter>,
}

impl Sheets {
    pub fn new(painter: Arc<Painter>, command_buffer: &mut CommandBuffer) -> Result<Self, String> {
        unsafe {
            // Swapchain creation
            let surface_instance = &painter.surface_instance;
            let physical_device = painter.physical_device;
            let surface = painter.surface;
            let surface_formats = surface_instance
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(|e| format!("at surface formats: {e}"))?;

            let surface_caps = surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(|e| format!("at surface capabilities: {e}"))?;

            let surface_present_modes = surface_instance
                .get_physical_device_surface_present_modes(physical_device, surface)
                .map_err(|e| format!("at surface present modes: {e}"))?;

            let surface_format = surface_formats
                .iter()
                .filter(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .filter(|format| {
                    let supported = painter
                        .instance
                        .get_physical_device_format_properties(
                            painter.physical_device,
                            format.format,
                        )
                        .optimal_tiling_features
                        .contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_DST
                                | vk::FormatFeatureFlags::STORAGE_IMAGE,
                        );
                    supported
                        && (format.format == vk::Format::B8G8R8A8_UNORM
                            || format.format == vk::Format::R8G8B8A8_UNORM
                            || format.format == vk::Format::B8G8R8A8_SRGB
                            || format.format == vk::Format::R8G8B8A8_SRGB)
                })
                .next()
                .cloned()
                .ok_or("no suitable surface format found".to_string())?;

            let mut surface_resolution = surface_caps.current_extent;
            if surface_resolution.width == u32::MAX || surface_resolution.height == u32::MAX {
                let window_res = painter.window.inner_size();
                surface_resolution.width = window_res.width;
                surface_resolution.height = window_res.height;
            }

            let surface_present_mode = surface_present_modes
                .iter()
                .filter(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
                .next()
                .cloned()
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_image_count = std::cmp::min(
                surface_caps.min_image_count + 1,
                if surface_caps.max_image_count == 0 {
                    std::u32::MAX
                } else {
                    surface_caps.max_image_count
                },
            );

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(painter.surface)
                .min_image_count(swapchain_image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(surface_resolution)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::STORAGE,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(surface_present_mode)
                .clipped(true);

            let swapchain_device = khr::swapchain::Device::new(&painter.instance, &painter.device);
            let swapchain = swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(|e| format!("at swapchain creation: {e}"))?;
            let swapchain_images = swapchain_device
                .get_swapchain_images(swapchain)
                .map_err(|e| format!("at swapchain images: {e}"))?
                .into_iter()
                .map(|image| {
                    let image_view =
                        Image2d::create_image_view(&painter, image, surface_format.format)
                            .map_err(|e| format!("at image view creation: {e}"))?;
                    Ok(Image2d {
                        image,
                        format: surface_format.format,
                        extent: surface_resolution,
                        is_swapchain_image: true,
                        painter: painter.clone(),
                        image_view,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            let commands = swapchain_images.iter().map(|image| GpuCommand::ImageAccessInit { image: image, access: ImageAccess::Present }).collect::<Vec<_>>();
            command_buffer.record(&commands, true)
                .map_err(|e| format!("at command buffer record: {e}"))?;
            let fence = CpuFuture::new(painter.clone(), false)
                .map_err(|e| format!("at fence creation: {e}"))?;
            command_buffer.submit(&[], &[], &[], Some(&fence))
                .map_err(|e| format!("at command buffer submit: {e}"))?;
            fence.wait().map_err(|e| format!("at fence wait: {e}"))?;
            command_buffer
                .reset()
                .map_err(|e| format!("at command buffer reset: {e}"))?;

            Ok(Self {
                swapchain_images,
                present_mode: surface_present_mode,
                surface_format,
                surface_resolution,
                swapchain,
                swapchain_device,
                painter,
            })
        }
    }

    pub fn refresh_resolution(&mut self, command_buffer: &mut CommandBuffer) -> Result<(), String> {
        unsafe {
            let surface_caps = self
                .painter
                .surface_instance
                .get_physical_device_surface_capabilities(
                    self.painter.physical_device,
                    self.painter.surface,
                )
                .map_err(|e| format!("at surface capabilities: {e}"))?;

            let new_resolution = surface_caps.current_extent;

            // Do not compare resolutions to avoid flickering in case of suboptimal swapchain
            // println!("new resolution: {:?}", new_resolution);
            // if new_resolution == self.surface_resolution {
            //     return Ok(false);
            // }

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.painter.surface)
                .min_image_count(self.swapchain_images.len() as u32)
                .image_format(self.surface_format.format)
                .image_color_space(self.surface_format.color_space)
                .image_extent(new_resolution)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::STORAGE,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(surface_caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(self.present_mode)
                .old_swapchain(self.swapchain)
                .clipped(true);

            let old_swapchain = self.swapchain;

            let new_swapchain = self
                .swapchain_device
                .create_swapchain(&swapchain_create_info, None)
                .map_err(|e| format!("at new swapchain creation: {e}"))?;

            let new_swapchain_images = self
                .swapchain_device
                .get_swapchain_images(new_swapchain)
                .map_err(|e| format!("at fetching swapchain images: {e}"))?
                .into_iter()
                .map(|image| {
                    let image_view = Image2d::create_image_view(
                        &self.painter,
                        image,
                        self.surface_format.format,
                    )
                    .map_err(|e| format!("at image view creation: {e}"))?;
                    Ok(Image2d {
                        image_view,
                        image,
                        format: self.surface_format.format,
                        extent: new_resolution,
                        is_swapchain_image: true,
                        painter: self.painter.clone(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            let commands = new_swapchain_images.iter().map(|image| GpuCommand::ImageAccessInit { image: image, access: ImageAccess::Present }).collect::<Vec<_>>();
            command_buffer.record(&commands, true)
                .map_err(|e| format!("at command buffer record: {e}"))?;
            let fence = CpuFuture::new(self.painter.clone(), false)
                .map_err(|e| format!("at fence creation: {e}"))?;
            command_buffer.submit(&[], &[], &[], Some(&fence))
                .map_err(|e| format!("at command buffer submit: {e}"))?;
            fence.wait().map_err(|e| format!("at fence wait: {e}"))?;
            command_buffer
                .reset()
                .map_err(|e| format!("at command buffer reset: {e}"))?;

            self.swapchain = new_swapchain;
            self.swapchain_images = new_swapchain_images;

            self.swapchain_device.destroy_swapchain(old_swapchain, None);

            self.surface_resolution = surface_caps.current_extent;
            Ok(())
        }
    }

    pub fn acquire_next_image(
        &mut self,
        semaphore: Option<&GpuFuture>,
        mut fence: Option<&CpuFuture>,
        command_buffer: &mut CommandBuffer
    ) -> Result<u32, String> {
        unsafe {
            let vk_fence = fence.map_or(vk::Fence::null(), |fence| fence.fence);
            let vk_semaphore =
                semaphore.map_or(vk::Semaphore::null(), |semaphore| semaphore.semaphore);
            if vk_fence == vk::Fence::null() && vk_semaphore == vk::Semaphore::null() {
                return Err("either fence or semaphore must be provided".to_string());
            }
            loop {
                let (img_id, refresh_needed) = match self
                    .swapchain_device
                    .acquire_next_image(self.swapchain, std::u64::MAX, vk_semaphore, vk_fence) {
                        Ok((i_id, ref_needed)) => (Some(i_id), ref_needed),
                        Err(e) => {
                            if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                                (None, true)
                            } else {
                                return Err(format!("at acquiring next image: {e}"))?;
                            }
                        }
                    };
                if refresh_needed {
                self.refresh_resolution(command_buffer)
                    .map_err(|e| format!("at refreshing swapchain resolution: {e}"))?;
                    fence.as_mut().map(|f| {
                        f.wait().map_err(|e| format!("at fence wait before refresh: {e}"))?;
                        f.reset().map_err(|e| format!("at fence reset before refresh: {e}"))?;
                        Ok::<(), String>(())
                    }).transpose()?;
                    continue;
                }
                if let Some(i_id) = img_id {
                    return Ok(i_id)
                }
            }
        }
    }

    pub fn present_image(
        &self,
        image_index: u32,
        wait_semaphores: &[&GpuFuture],
    ) -> Result<(), String> {
        unsafe {
            let wait_semaphores = wait_semaphores
                .iter()
                .map(|semaphore| semaphore.semaphore)
                .collect::<Vec<_>>();
            match self.swapchain_device
                .queue_present(
                    self.painter.graphics_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&wait_semaphores)
                        .swapchains(&[self.swapchain])
                        .image_indices(&[image_index]),
                ) {
                Ok(_) => Ok(()),
                Err(e) => {
                    if e != vk::Result::ERROR_OUT_OF_DATE_KHR {
                        Err(format!("at presenting image: {e}"))
                    } else {
                        Ok(())
                    }
                }
            }
        }
    }
}

impl Drop for Sheets {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
