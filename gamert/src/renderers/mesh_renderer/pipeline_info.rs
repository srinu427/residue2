use include_bytes_aligned::include_bytes_aligned;
use ash::vk;
use thiserror::Error;

static VERTEX_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "../shaders/mesh_painter.vert.spv");
static FRAGMENT_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "../shaders/mesh_painter.frag.spv");

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
pub enum PipelineInfoInitError {
    #[error("No compatible depth format found")]
    NoCompatibleDepthFormat,
    #[error("Error during render pass creation: {0}")]
    RenderPassCreationError(vk::Result),
    #[error("Error during descriptor set layout creation: {0}")]
    DescriptorSetLayoutCreationError(vk::Result),
    #[error("Error during pipeline layout creation: {0}")]
    PipelineLayoutCreationError(vk::Result),
    #[error("Error during pipeline creation: {0}")]
    PipelineCreationError(vk::Result),
    #[error("Error during texture sampler creation: {0}")]
    TextureSamplerCreationError(vk::Result),
    #[error("Error during shader module ({0}) creation: {1}")]
    ShaderModuleCreationError(&'static str, vk::Result),
    #[error("Error during vertex buffer creation: {0}")]
    VertexBufferCreationError(vk::Result),
    #[error("Error during index buffer creation: {0}")]
    IndexBufferCreationError(vk::Result),
}

pub struct PipelineInfo {
    pub(crate) depth_format: vk::Format,
    pub(crate) render_pass: vk::RenderPass,
    pub(crate) descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) sampler: vk::Sampler,
}

impl PipelineInfo {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Result<Self, PipelineInfoInitError> {
        // Select depth format
        let depth_format = find_depth_format(instance, physical_device)
            .ok_or(PipelineInfoInitError::NoCompatibleDepthFormat)?;

        unsafe {
            // Render pass creation
            let render_pass = device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&[
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
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE),
                        ])
                        .subpasses(&[
                            vk::SubpassDescription::default()
                                .color_attachments(&[vk::AttachmentReference::default()
                                    .attachment(0)
                                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
                                .depth_stencil_attachment(&vk::AttachmentReference::default()
                                    .attachment(1)
                                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
                                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS),
                        ]),
                    None
                )
                    .map_err(PipelineInfoInitError::RenderPassCreationError)?;
            
            // Descriptor set layout 1:
            // Binding 0: Vertex Data
            // Binding 1: Index Data
            // Binding 2: Scene Data
            // Binding 3: Texture Samplers

            let scene_descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&[
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(0)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(2)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .descriptor_count(1)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                    ]),
                    None
                )
                    .map_err(PipelineInfoInitError::DescriptorSetLayoutCreationError)?;

            // Descriptor set layout 2
            // Binding 0: Texture Data

            let texture_descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&[vk::DescriptorSetLayoutBinding::default()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(100)
                            .stage_flags(vk::ShaderStageFlags::ALL)]),
                    None
                )
                    .map_err(PipelineInfoInitError::DescriptorSetLayoutCreationError)?;

            // Pipeline layout creation
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[scene_descriptor_set_layout, texture_descriptor_set_layout]),
                    None
                )
                .map_err(PipelineInfoInitError::PipelineLayoutCreationError)?;

            // Vertex shader module creation
            let vertex_shader_module = device
                    .create_shader_module(
                        &vk::ShaderModuleCreateInfo::default()
                            .code(VERTEX_SHADER_CODE.align_to::<u32>().1)
                            .flags(vk::ShaderModuleCreateFlags::empty()),
                        None,
                    )
                    .map_err(|e| PipelineInfoInitError::ShaderModuleCreationError("Vertex", e))?;
            // Fragment shader module creation
            let fragment_shader_module = device
                    .create_shader_module(
                        &vk::ShaderModuleCreateInfo::default()
                            .code(FRAGMENT_SHADER_CODE.align_to::<u32>().1)
                            .flags(vk::ShaderModuleCreateFlags::empty()),
                        None,
                    )
                    .map_err(|e| PipelineInfoInitError::ShaderModuleCreationError("Fragment", e))?;

            // Graphics pipeline creation
            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(vertex_shader_module)
                                .name(c"main"),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(fragment_shader_module)
                                .name(c"main"),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(&vk::PipelineInputAssemblyStateCreateInfo::default()
                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST))
                        .dynamic_state(&vk::PipelineDynamicStateCreateInfo::default()
                            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]))
                        .viewport_state(&vk::PipelineViewportStateCreateInfo::default()
                            .viewport_count(1)
                            .scissor_count(1))
                        .rasterization_state(&vk::PipelineRasterizationStateCreateInfo::default()
                            .polygon_mode(vk::PolygonMode::FILL)
                            .cull_mode(vk::CullModeFlags::BACK)
                            .front_face(vk::FrontFace::COUNTER_CLOCKWISE))
                        .multisample_state(&vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1))
                        .depth_stencil_state(&vk::PipelineDepthStencilStateCreateInfo::default()
                            .depth_test_enable(true)
                            .depth_write_enable(true)
                            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                            .depth_bounds_test_enable(false)
                            .stencil_test_enable(false))
                        .color_blend_state(&vk::PipelineColorBlendStateCreateInfo::default()
                            .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                .color_write_mask(vk::ColorComponentFlags::RGBA)
                                .blend_enable(false)]))
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0)],
                    None,
                )
                .map_err(|(_, e)| PipelineInfoInitError::PipelineCreationError(e))?
                .get(0)
                .cloned()
                .ok_or(PipelineInfoInitError::PipelineCreationError(vk::Result::ERROR_UNKNOWN))?;

            // Cleanup shader modules
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);

            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT),
                    None
                )
                    .map_err(PipelineInfoInitError::TextureSamplerCreationError)?;
            
            Ok(Self {
                depth_format,
                render_pass,
                descriptor_set_layouts: vec![scene_descriptor_set_layout, texture_descriptor_set_layout],
                pipeline,
                pipeline_layout,
                sampler,
            })
        }
    }

    pub fn cleanup(&self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            for &layout in &self.descriptor_set_layouts {
                device.destroy_descriptor_set_layout(layout, None);
            }
        }
    }
}
