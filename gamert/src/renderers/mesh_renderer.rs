use ash::vk;
use thiserror::Error;

static DEPTH_FORMAT_PREFERENCE_LIST: &[vk::Format] = &[
    vk::Format::D24_UNORM_S8_UINT,
    vk::Format::D32_SFLOAT,
    vk::Format::D32_SFLOAT_S8_UINT,
    vk::Format::D16_UNORM,
];

fn find_depth_format(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<vk::Format>{
    DEPTH_FORMAT_PREFERENCE_LIST
        .iter()
        .find_map(|&format| {
            let format_properties = unsafe {
                instance.get_physical_device_format_properties(physical_device, format)
            };
            if format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                Some(format)
            } else {
                None
            }
        })
}

#[derive(Debug, Error)]
pub enum MeshRendererInitError {
    #[error("No compatible depth format found")]
    NoCompatibleDepthFormat,
    #[error("Error during render pass creation: {0}")]
    RenderPassCreationError(vk::Result),
    #[error("Error during descriptor set layout creation: {0}")]
    DescriptorSetLayoutCreationError(vk::Result),
}

pub struct MeshRenderer {
    depth_format: vk::Format,
    render_pass: vk::RenderPass,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    texture_sampler: vk::Sampler,
}

impl MeshRenderer {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        res: vk::Extent2D,
    ) -> Result<Self, MeshRendererInitError> {
        // Select depth format
        let depth_format = find_depth_format(instance, physical_device)
            .ok_or(MeshRendererInitError::NoCompatibleDepthFormat)?;

        let attachments = vec![
            vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE),
            vk::AttachmentDescription::default()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        ];

        let subpass_color_attachments = [
            vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];
        let subpass_depth_attachment = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass_info = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&subpass_color_attachments)
            .depth_stencil_attachment(&subpass_depth_attachment);

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&[subpass_info]);
        let render_pass = unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .map_err(MeshRendererInitError::RenderPassCreationError)?
        };

        // Descriptor set layout 1:
        // Binding 0: Scene Data
        // Binding 1: Texture Samplers

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&[
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::ALL),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::ALL),
            ]);
        
        let scene_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .map_err(MeshRendererInitError::DescriptorSetLayoutCreationError)?
        };


        // Descriptor set layout 2
        // Binding 0: Texture Data
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&[vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(100)
                .stage_flags(vk::ShaderStageFlags::ALL)]);
        
        let texture_descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .map_err(MeshRendererInitError::DescriptorSetLayoutCreationError)?
        };

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&[scene_descriptor_set_layout, texture_descriptor_set_layout])
            ;
    }
}
