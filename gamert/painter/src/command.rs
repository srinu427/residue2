use std::sync::Arc;

use ash::vk;
use hashbrown::HashMap;

use crate::{CpuFuture, GpuFuture, Image2d, ImageAccess, Painter, image::is_format_depth};

pub struct ImageTransitionInfo<'a> {
    pub image: &'a Image2d,
    pub old_access: Option<ImageAccess>,
    pub new_access: Option<ImageAccess>,
}

pub enum GpuRenderPassCommand {
    BindPipeline {
        pipeline: usize,
    },
    BindShaderInput {
        pipeline_layout: usize,
        descriptor_sets: Vec<vk::DescriptorSet>,
    },
    BindVertexBuffers {
        buffers: Vec<vk::Buffer>,
    },
    BindIndexBuffer {
        buffer: vk::Buffer,
    },
    SetPushConstant {
        pipeline_layout: usize,
        data: Vec<u8>,
    },
    Draw {
        count: u32,
        vertex_offset: i32,
        index_offset: u32,
    },
}

pub enum GpuCommand<'a> {
    ImageAccessInit {
        image: &'a Image2d,
        access: ImageAccess,
    },
    ImageAccessHint {
        image: &'a Image2d,
        access: ImageAccess,
    },
    BlitFullImage {
        src: &'a Image2d,
        dst: &'a Image2d,
    },
    RunRenderPass {
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
        clear_values: Vec<vk::ClearValue>,
        pipelines: Vec<vk::Pipeline>,
        pipeline_layouts: Vec<vk::PipelineLayout>,
        commands: Vec<GpuRenderPassCommand>,
    },
    CopyBufferToImageComplete {
        buffer: vk::Buffer,
        image: &'a Image2d,
    },
}

impl<'a> GpuCommand<'a> {
    pub fn access_transitions(&self) -> Vec<ImageTransitionInfo> {
        match self {
            Self::ImageAccessInit { image, access } => vec![ImageTransitionInfo {
                image,
                old_access: Some(ImageAccess::None),
                new_access: Some(*access),
            }],
            Self::ImageAccessHint { image, access } => vec![ImageTransitionInfo {
                image,
                old_access: None,
                new_access: Some(*access),
            }],
            Self::BlitFullImage { src, dst } => vec![
                ImageTransitionInfo {
                    image: src,
                    old_access: None,
                    new_access: Some(ImageAccess::TransferRead),
                },
                ImageTransitionInfo {
                    image: dst,
                    old_access: None,
                    new_access: Some(ImageAccess::TransferWrite),
                },
            ],
            Self::RunRenderPass {
                render_pass: _,
                framebuffer: _,
                extent: _,
                clear_values: _,
                pipelines: _,
                pipeline_layouts: _,
                commands: _,
            } => vec![],
            Self::CopyBufferToImageComplete { buffer: _, image } => vec![ImageTransitionInfo {
                image,
                old_access: None,
                new_access: Some(ImageAccess::TransferWrite),
            }],
        }
    }
}

pub struct CommandBuffer {
    pub command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    painter: Arc<Painter>,
}

impl CommandBuffer {
    pub fn submit(
        &self,
        signal_semaphores: &[&GpuFuture],
        wait_semaphores: &[&GpuFuture],
        wait_stages: &[vk::PipelineStageFlags],
        fence: Option<&CpuFuture>,
    ) -> Result<(), String> {
        unsafe {
            let vk_fence = fence.map_or(vk::Fence::null(), |fence| fence.fence);
            let signal_semaphores = signal_semaphores
                .iter()
                .map(|semaphore| semaphore.semaphore)
                .collect::<Vec<_>>();
            let wait_semaphores = wait_semaphores
                .iter()
                .map(|semaphore| semaphore.semaphore)
                .collect::<Vec<_>>();
            self.painter
                .device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default()
                        .signal_semaphores(&signal_semaphores)
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(wait_stages)
                        .command_buffers(&[self.command_buffer])],
                    vk_fence,
                )
                .map_err(|e| format!("at queue submit: {e}"))?;
        }
        Ok(())
    }

    pub fn reset(&self) -> Result<(), String> {
        unsafe {
            self.painter
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("at command buffer reset: {e}"))?;
        }
        Ok(())
    }

    pub fn begin(&self, one_time: bool) -> Result<(), String> {
        unsafe {
            let flags = if one_time {
                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
            } else {
                vk::CommandBufferUsageFlags::empty()
            };
            self.painter
                .device
                .begin_command_buffer(
                    self.command_buffer,
                    &vk::CommandBufferBeginInfo::default().flags(flags),
                )
                .map_err(|e| format!("at command buffer begin: {e}"))?;
        }
        Ok(())
    }

    pub fn end(&self) -> Result<(), String> {
        unsafe {
            self.painter
                .device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| format!("at command buffer end: {e}"))?;
        }
        Ok(())
    }

    pub fn record(&mut self, commands: &[GpuCommand], one_time: bool) -> Result<(), String> {
        let begin_flags = if one_time {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(begin_flags);
        unsafe {
            self.painter
                .device
                .begin_command_buffer(self.command_buffer, &command_buffer_begin_info)
                .map_err(|e| format!("at command buffer begin: {e}"))?;

            let mut image_accesses = HashMap::new();

            for (command_idx, command) in commands.iter().enumerate() {
                for transition in command.access_transitions() {
                    let (_, image_transitions) = image_accesses
                        .entry(transition.image.image)
                        .or_insert((transition.image, vec![]));
                    match transition.old_access {
                        Some(old_access) => {
                            if image_transitions.len() == 0 {
                                image_transitions.push((command_idx, old_access));
                            }
                        }
                        None => {}
                    }
                    match transition.new_access {
                        Some(new_access) => {
                            if let Some((_, last_access)) = image_transitions.last() {
                                if *last_access != new_access {
                                    image_transitions.push((command_idx + 1, new_access));
                                }
                            } else {
                                image_transitions.push((command_idx + 1, new_access));
                            }
                        }
                        None => {}
                    }
                }
            }

            for (command_idx, command) in commands.iter().enumerate() {
                for (_, (image, transitions_needed)) in image_accesses.iter() {
                    let Some(access_idx) = transitions_needed
                        .iter()
                        .position(|(x, _)| *x == command_idx + 1)
                    else {
                        continue;
                    };
                    let access_new = transitions_needed[access_idx].1;
                    if access_idx == 0 {
                        continue;
                    }
                    let access_old = transitions_needed[access_idx - 1].1;
                    if access_old == access_new {
                        continue;
                    }
                    let is_depth_image = is_format_depth(image.format);
                    // println!("image transition: {:?} {access_old:?} -> {access_new:?}", image.image);
                    self.painter.device.cmd_pipeline_barrier(
                        self.command_buffer,
                        access_old.get_pipeline_stage(),
                        access_new.get_pipeline_stage(),
                        vk::DependencyFlags::BY_REGION,
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::default()
                            .image(image.image)
                            .src_access_mask(access_old.to_access_flags(is_depth_image))
                            .dst_access_mask(access_new.to_access_flags(is_depth_image))
                            .old_layout(access_old.get_image_layout(is_depth_image))
                            .new_layout(access_new.get_image_layout(is_depth_image))
                            .subresource_range(image.get_subresource_range())],
                    );
                }
                match command {
                    GpuCommand::ImageAccessInit { image, access } => {}
                    GpuCommand::ImageAccessHint { image, access } => {}
                    GpuCommand::BlitFullImage { src, dst } => {
                        self.painter.device.cmd_blit_image(
                            self.command_buffer,
                            src.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            dst.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::ImageBlit::default()
                                .src_subresource(src.get_subresource_layers())
                                .dst_subresource(dst.get_subresource_layers())
                                .src_offsets(src.get_full_size_offset())
                                .dst_offsets(dst.get_full_size_offset())],
                            vk::Filter::NEAREST,
                        );
                    }
                    GpuCommand::RunRenderPass {
                        render_pass,
                        framebuffer,
                        extent,
                        clear_values,
                        pipelines,
                        pipeline_layouts,
                        commands: rp_commands,
                    } => {
                        self.painter.device.cmd_begin_render_pass(
                            self.command_buffer,
                            &vk::RenderPassBeginInfo::default()
                                .render_pass(*render_pass)
                                .framebuffer(*framebuffer)
                                .render_area(vk::Rect2D::default().extent(*extent))
                                .clear_values(clear_values),
                            vk::SubpassContents::INLINE,
                        );
                        self.painter.device.cmd_set_viewport(
                            self.command_buffer,
                            0,
                            &[vk::Viewport::default()
                                .width(extent.width as f32)
                                .height(extent.height as f32)],
                        );
                        self.painter.device.cmd_set_scissor(
                            self.command_buffer,
                            0,
                            &[vk::Rect2D::default().extent(*extent)],
                        );

                        for rp_command in rp_commands.iter() {
                            match rp_command {
                                GpuRenderPassCommand::BindPipeline { pipeline } => {
                                    self.painter.device.cmd_bind_pipeline(
                                        self.command_buffer,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        pipelines[*pipeline],
                                    );
                                }
                                GpuRenderPassCommand::BindShaderInput {
                                    pipeline_layout,
                                    descriptor_sets,
                                } => {
                                    self.painter.device.cmd_bind_descriptor_sets(
                                        self.command_buffer,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        pipeline_layouts[*pipeline_layout],
                                        0,
                                        descriptor_sets,
                                        &[],
                                    );
                                }
                                GpuRenderPassCommand::BindVertexBuffers { buffers } => {
                                    self.painter.device.cmd_bind_vertex_buffers(
                                        self.command_buffer,
                                        0,
                                        buffers,
                                        &[0],
                                    );
                                }
                                GpuRenderPassCommand::BindIndexBuffer { buffer } => {
                                    self.painter.device.cmd_bind_index_buffer(
                                        self.command_buffer,
                                        *buffer,
                                        0,
                                        vk::IndexType::UINT32,
                                    );
                                }
                                GpuRenderPassCommand::SetPushConstant {
                                    pipeline_layout,
                                    data,
                                } => {
                                    self.painter.device.cmd_push_constants(
                                        self.command_buffer,
                                        pipeline_layouts[*pipeline_layout],
                                        vk::ShaderStageFlags::ALL,
                                        0,
                                        data,
                                    );
                                }
                                GpuRenderPassCommand::Draw {
                                    count,
                                    vertex_offset,
                                    index_offset,
                                } => {
                                    self.painter.device.cmd_draw_indexed(
                                        self.command_buffer,
                                        *count,
                                        1,
                                        *index_offset,
                                        *vertex_offset,
                                        0,
                                    );
                                }
                            }
                        }

                        self.painter.device.cmd_end_render_pass(self.command_buffer);
                    }
                    GpuCommand::CopyBufferToImageComplete { buffer, image } => {
                        self.painter.device.cmd_copy_buffer_to_image(
                            self.command_buffer,
                            *buffer,
                            image.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[vk::BufferImageCopy::default()
                                .buffer_offset(0)
                                .buffer_row_length(0)
                                .buffer_image_height(0)
                                .image_subresource(image.get_subresource_layers())
                                .image_offset(vk::Offset3D::default())
                                .image_extent(image.extent3d())],
                        );
                    }
                }
            }

            self.painter
                .device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| format!("at command buffer end: {e}"))?;
        }
        Ok(())
    }
}

pub struct CommandPool {
    pub command_pool: vk::CommandPool,
    queue: vk::Queue,
    painter: Arc<Painter>,
}

impl CommandPool {
    pub fn new(painter: Arc<Painter>) -> Result<Self, String> {
        unsafe {
            let command_pool = painter
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(painter.graphics_queue_family_index),
                    None,
                )
                .map_err(|e| format!("at command pool creation: {e}"))?;
            Ok(Self {
                command_pool,
                painter: painter.clone(),
                queue: painter.graphics_queue,
            })
        }
    }

    pub fn allocate_command_buffers(&self, count: usize) -> Result<Vec<CommandBuffer>, String> {
        unsafe {
            let command_buffers = self
                .painter
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(count as u32),
                )
                .map_err(|e| format!("at command buffer allocation: {e}"))?
                .into_iter()
                .map(|command_buffer| CommandBuffer {
                    command_buffer,
                    command_pool: self.command_pool,
                    queue: self.queue,
                    painter: self.painter.clone(),
                })
                .collect();
            Ok(command_buffers)
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
