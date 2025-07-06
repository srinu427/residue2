use std::sync::Arc;

use ash::vk;

use crate::{Image2d, Painter, ShaderInputBindingInfo, ShaderInputLayout, ShaderModule};

pub struct SingePassRenderPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub render_pass: vk::RenderPass,
    pub shader_input_layouts: Vec<ShaderInputLayout>,
    pub push_constant_size: usize,
    pub painter: Arc<Painter>,
}

impl SingePassRenderPipeline {
    pub fn new(
        painter: Arc<Painter>,
        color_attachments: Vec<(vk::Format, vk::AttachmentLoadOp, vk::AttachmentStoreOp)>,
        depth_attachment: Option<(vk::Format, vk::AttachmentLoadOp, vk::AttachmentStoreOp)>,
        input_layouts: Vec<Vec<ShaderInputBindingInfo>>,
        push_constant_size: usize,
        vertex_shader_code: &[u8],
        fragment_shader_code: &[u8],
        vertex_binding_descriptions: Vec<vk::VertexInputBindingDescription>,
        vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription>,
    ) -> Result<Self, String> {
        let color_attachments = color_attachments
            .iter()
            .map(|(format, load_op, store_op)| {
                vk::AttachmentDescription::default()
                    .format(*format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(*load_op)
                    .store_op(*store_op)
                    .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            })
            .collect::<Vec<_>>();
        let depth_attachment = depth_attachment.map(|(format, load_op, store_op)| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(load_op)
                .store_op(store_op)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        });
        let all_attchments = if let Some(depth_attachment) = depth_attachment.clone() {
            let mut a = color_attachments.clone();
            a.append(&mut vec![depth_attachment]);
            a
        } else {
            color_attachments.clone()
        };
        let subpass_color_attachments = (0..color_attachments.len())
            .map(|i| {
                vk::AttachmentReference::default()
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .attachment(i as _)
            })
            .collect::<Vec<_>>();
        let subpass_depth_attachment = depth_attachment.map(|_| {
            vk::AttachmentReference::default()
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .attachment(color_attachments.len() as _)
        });
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&subpass_color_attachments);
        match subpass_depth_attachment.as_ref() {
            Some(subpass_depth_attachment) => {
                subpass = subpass.depth_stencil_attachment(subpass_depth_attachment);
            }
            None => {}
        }
        let subpass = [subpass];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&all_attchments)
            .subpasses(&subpass);
        let render_pass = unsafe {
            painter
                .device
                .create_render_pass(&render_pass_create_info, None)
                .map_err(|e| format!("at render pass creation: {e}"))?
        };

        let shader_input_layouts = input_layouts
            .iter()
            .map(|input_layout| ShaderInputLayout::new(painter.clone(), input_layout.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let set_layouts = shader_input_layouts
            .iter()
            .map(|input_layout| input_layout.descriptor_set_layout)
            .collect::<Vec<_>>();
        let pc_ranges = if push_constant_size > 0 {
            vec![
                vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .offset(0)
                    .size(push_constant_size as u32),
            ]
        } else {
            vec![]
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_ranges);
        let pipeline_layout = unsafe {
            painter
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| format!("at pipeline layout creation: {e}"))?
        };
        let pipeline = unsafe {
            let vertex_shader_module = ShaderModule::new(painter.clone(), &vertex_shader_code)?;
            let fragment_shader_module = ShaderModule::new(painter.clone(), &fragment_shader_code)?;
            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(*vertex_shader_module.get_vk())
                    .name(c"main")
                    .into(),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(*fragment_shader_module.get_vk())
                    .name(c"main")
                    .into(),
            ];
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&vertex_binding_descriptions)
                .vertex_attribute_descriptions(&vertex_attribute_descriptions);
            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_blend_attachments);
            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(depth_attachment.is_some())
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false);
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
                .render_pass(render_pass)
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .layout(pipeline_layout)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .depth_stencil_state(&depth_stencil_state)
                .dynamic_state(&dynamic_state);
            painter
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|(_, e)| format!("at pipeline creation: {e}"))?
                .swap_remove(0)
        };
        Ok(Self {
            render_pass,
            shader_input_layouts,
            push_constant_size,
            pipeline_layout,
            pipeline,
            painter,
        })
    }

    pub fn create_render_output(&self, attachments: Vec<&Image2d>) -> Result<RenderOutput, String> {
        unsafe {
            let attachment_views = attachments
                .iter()
                .map(|image| image.image_view)
                .collect::<Vec<_>>();
            let framebuffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(&attachment_views)
                .width(attachments[0].extent.width)
                .height(attachments[0].extent.height)
                .layers(1);
            let framebuffer = self
                .painter
                .device
                .create_framebuffer(&framebuffer_create_info, None)
                .map_err(|e| format!("at framebuffer creation: {e}"))?;
            Ok(RenderOutput {
                extent: attachments[0].extent,
                render_pass: self.render_pass,
                framebuffer,
                painter: self.painter.clone(),
            })
        }
    }
}

impl Drop for SingePassRenderPipeline {
    fn drop(&mut self) {
        unsafe {
            self.painter.device.destroy_pipeline(self.pipeline, None);
            self.painter
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.painter
                .device
                .destroy_render_pass(self.render_pass, None);
        }
    }
}

pub struct RenderOutput {
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    painter: Arc<Painter>,
}

impl Drop for RenderOutput {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .destroy_framebuffer(self.framebuffer, None);
        }
    }
}
