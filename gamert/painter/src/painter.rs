use ash::{ext, khr, vk};
use strum::{Display, EnumCount};
use thiserror::Error;
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

static DEPTH_FORMAT_PREFERENCE_LIST: &[vk::Format] = &[
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D32_SFLOAT,
    vk::Format::D32_SFLOAT_S8_UINT,
    vk::Format::D16_UNORM,
];

pub fn get_instance_layers() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        c"VK_LAYER_KHRONOS_validation".as_ptr(),
    ]
}

pub fn get_instance_extensions() -> Vec<*const i8> {
    vec![
        #[cfg(debug_assertions)]
        ext::debug_utils::NAME.as_ptr(),
        khr::get_physical_device_properties2::NAME.as_ptr(),
        khr::surface::NAME.as_ptr(),
        #[cfg(target_os = "windows")]
        khr::win32_surface::NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        khr::xlib_surface::NAME.as_ptr(),
        #[cfg(target_os = "linux")]
        khr::wayland_surface::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_enumeration::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        ext::metal_surface::NAME.as_ptr(),
        #[cfg(target_os = "android")]
        khr::android_surface::NAME.as_ptr(),
    ]
}

pub fn get_device_extensions() -> Vec<*const i8> {
    vec![
        khr::swapchain::NAME.as_ptr(),
        ext::descriptor_indexing::NAME.as_ptr(),
        khr::dynamic_rendering::NAME.as_ptr(),
        #[cfg(target_os = "macos")]
        khr::portability_subset::NAME.as_ptr(),
    ]
}

pub fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, PainterError> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Residue VK App")
        .application_version(0)
        .engine_name(c"Residue Engine")
        .engine_version(0)
        .api_version(vk::API_VERSION_1_2);

    let layers = get_instance_layers();
    let extensions = get_instance_extensions();

    #[cfg(target_os = "macos")]
    let vk_instance_create_info = vk::InstanceCreateInfo::default()
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    #[cfg(not(target_os = "macos"))]
    let vk_instance_create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);

    unsafe {
        entry
            .create_instance(&vk_instance_create_info, None)
            .map_err(PainterError::VkInstanceError)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, strum::EnumCount, Display)]
#[repr(usize)]
pub enum ImageFormatType {
    Rgba8Unorm = 0,
    DepthStencilOptimal = 1,
}

#[derive(Error, Debug)]
pub enum PainterError{
    #[error("Error loading Vulkan: {0}")]
    VkLoadError(ash::LoadingError),
    #[error("Error creating a Vulkan Instance: {0}")]
    VkInstanceError(vk::Result),
    #[error("Error getting raw display handle: {0}")]
    GetRawDisplayHandleError(winit::raw_window_handle::HandleError),
    #[error("Error getting raw window handle: {0}")]
    GetRawWindowHandleError(winit::raw_window_handle::HandleError),
    #[error("Error creating surface: {0}")]
    SurfaceCreationError(vk::Result),
    #[error("Error getting GPU list: {0}")]
    GetGpusError(vk::Result),
    #[error("No Supported GPUs found")]
    NoSupportedGpu,
    #[error("Error creating Vulkan Logical Device: {0}")]
    LogicalDeviceCreateError(vk::Result),
    #[error("Can't finding suitable image format of type: {0}")]
    NoSuitableImageFormat(ImageFormatType),
    #[error("Error creating allocation error: {0}")]
    UnableToCreateAllocator(gpu_allocator::AllocationError),
}

pub struct Painter {
    pub image_formats: [vk::Format; ImageFormatType::COUNT],
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub surface: vk::SurfaceKHR,
    pub surface_instance: khr::surface::Instance,
    pub instance: ash::Instance,
    pub entry: ash::Entry,
    pub window: Window,
}

impl Painter {
    fn select_gpu_queue(
        instance: &ash::Instance,
        surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Option<u32> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        queue_families
            .iter()
            .enumerate()
            .filter(|(i, queue_family)| {
                let supports_graphics = queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_present = unsafe {
                    surface_instance
                        .get_physical_device_surface_support(physical_device, *i as u32, surface)
                        .unwrap_or(false)
                };
                supports_graphics && supports_present
            })
            .max_by_key(|(_, queue_family)| queue_family.queue_count)
            .map(|(i, _)| i as u32)
    }

    pub fn new(window: Window) -> Result<Self, PainterError> {
        unsafe {
            let entry = ash::Entry::load().map_err(PainterError::VkLoadError)?;

            let instance = create_instance(&entry)?;

            let surface_instance = khr::surface::Instance::new(&entry, &instance);

            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window
                    .display_handle()
                    .map_err(PainterError::GetRawDisplayHandleError)?
                    .as_raw(),
                window
                    .window_handle()
                    .map_err(PainterError::GetRawWindowHandleError)?
                    .as_raw(),
                None,
            )
            .map_err(PainterError::SurfaceCreationError)?;

            let mut physical_devices = instance
                .enumerate_physical_devices()
                .map_err(PainterError::GetGpusError)?
                .iter()
                .filter_map(|&physical_device| {
                    Self::select_gpu_queue(&instance, &surface_instance, physical_device, surface)
                        .map(|queue_family_index| (physical_device, queue_family_index))
                })
                .collect::<Vec<_>>();

            physical_devices.sort_by_key(|(physical_device, _)| {
                let is_dedicated = instance
                    .get_physical_device_properties(*physical_device)
                    .device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU;
                if is_dedicated { 2 } else { 1 }
            });

            let (physical_device, graphics_queue_family_index) =
                physical_devices
                    .last()
                    .cloned()
                    .ok_or(PainterError::NoSupportedGpu)?;

            let queue_priorities = [1.0];
            let queue_infos = vec![
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(graphics_queue_family_index)
                    .queue_priorities(&queue_priorities),
            ];

            let device_extensions = get_device_extensions();

            let mut device_12_features = vk::PhysicalDeviceVulkan12Features::default()
                .descriptor_indexing(true)
                .runtime_descriptor_array(true)
                .descriptor_binding_sampled_image_update_after_bind(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_variable_descriptor_count(true);
            let mut dynamic_rendering_switch =
                vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
            let device_features = vk::PhysicalDeviceFeatures::default();

            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extensions)
                .enabled_features(&device_features)
                .push_next(&mut device_12_features)
                .push_next(&mut dynamic_rendering_switch);

            let device = instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(PainterError::LogicalDeviceCreateError)?;

            let graphics_queue = device.get_device_queue(graphics_queue_family_index, 0);

            let rgba8_format = vk::Format::R8G8B8A8_UNORM;
            let depth_format = DEPTH_FORMAT_PREFERENCE_LIST
                .iter()
                .find_map(|&format| {
                    let format_properties = instance.get_physical_device_format_properties(
                        physical_device,
                        format,
                    );
                    if format_properties.optimal_tiling_features.contains(
                        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
                    ) {
                        Some(format)
                    } else {
                        None
                    }
                })
                .ok_or(PainterError::NoSuitableImageFormat(ImageFormatType::DepthStencilOptimal))?;
            
            let mut image_formats = [vk::Format::UNDEFINED; ImageFormatType::COUNT];
            image_formats[ImageFormatType::Rgba8Unorm as usize] = rgba8_format;
            image_formats[ImageFormatType::DepthStencilOptimal as usize] = depth_format;

            Ok(Self {
                instance,
                entry,
                surface_instance,
                surface,
                window,
                device,
                graphics_queue,
                graphics_queue_family_index,
                physical_device,
                image_formats,
            })
        }
    }

    pub fn new_allocator(&self) -> Result<gpu_allocator::vulkan::Allocator, PainterError> {
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: self.instance.clone(),
            device: self.device.clone(),
            physical_device: self.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(PainterError::UnableToCreateAllocator)
    }
}

impl Drop for Painter {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
