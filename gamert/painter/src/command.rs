use ash::vk;
use crossbeam::channel::Sender;
use hashbrown::HashMap;
use thiserror::Error;

use crate::{
    Buffer, CpuFuture, GpuFuture, Image2d, ImageAccess, Painter, RenderOutput,
    image::is_format_depth, painter::PainterDelete,
};

pub struct ImageTransitionInfo<'a> {
    pub image: &'a Image2d,
    pub old_access: Option<ImageAccess>,
    pub new_access: Option<ImageAccess>,
}

pub enum GpuRenderPassCommand<'a> {
    BindPipeline {
        pipeline: usize,
    },
    BindShaderInput {
        pipeline_layout: usize,
        descriptor_sets: Vec<vk::DescriptorSet>,
    },
    BindVertexBuffers {
        buffers: Vec<&'a Buffer>,
    },
    BindIndexBuffer {
        buffer: &'a Buffer,
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

impl<'a> GpuRenderPassCommand<'a> {
    pub fn apply_command(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        pipelines: &[vk::Pipeline],
        pipeline_layouts: &[vk::PipelineLayout],
    ) {
        unsafe {
            match self {
                GpuRenderPassCommand::BindPipeline { pipeline } => {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipelines[*pipeline],
                    );
                }
                GpuRenderPassCommand::BindShaderInput {
                    pipeline_layout,
                    descriptor_sets,
                } => {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layouts[*pipeline_layout],
                        0,
                        descriptor_sets,
                        &[],
                    );
                }
                GpuRenderPassCommand::BindVertexBuffers { buffers } => {
                    let buffers = buffers
                        .iter()
                        .map(|buffer| buffer.buffer)
                        .collect::<Vec<_>>();
                    let offsets = vec![0; buffers.len()];
                    device.cmd_bind_vertex_buffers(
                        command_buffer,
                        0,
                        &buffers,
                        &offsets,
                    );
                }
                GpuRenderPassCommand::BindIndexBuffer { buffer } => {
                    device.cmd_bind_index_buffer(
                        command_buffer,
                        buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                }
                GpuRenderPassCommand::SetPushConstant {
                    pipeline_layout,
                    data,
                } => {
                    device.cmd_push_constants(
                        command_buffer,
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
                    device.cmd_draw_indexed(
                        command_buffer,
                        *count,
                        1,
                        *index_offset,
                        *vertex_offset,
                        0,
                    );
                }
            }
        }
    }
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
        render_output: &'a RenderOutput,
        clear_values: Vec<vk::ClearValue>,
        pipelines: Vec<vk::Pipeline>,
        pipeline_layouts: Vec<vk::PipelineLayout>,
        commands: Vec<GpuRenderPassCommand<'a>>,
    },
    CopyBufferToImageComplete {
        buffer: &'a Buffer,
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
                render_output: _,
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

#[derive(Debug, Error)]
pub enum CommandBufferError {
    #[error("Error beginning command buffer recording: {0}")]
    BeginError(vk::Result),
    #[error("Error ending command buffer recording: {0}")]
    EndError(vk::Result),
    #[error("Error resetting command buffer: {0}")]
    ResetError(vk::Result),
}

pub struct CommandBuffer {
    pub command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
}

impl CommandBuffer {
    
}

#[derive(Debug, Error)]
pub enum CommandPoolError {
    #[error("Error at creating Vulkan command pool: {0}")]
    CreateError(vk::Result),
    #[error("Error allocating Vulkan Command Buffers: {0}")]
    CommandBufferAllocationError(vk::Result),
}

pub struct CommandPool {
    pub command_pool: vk::CommandPool,
    queue: vk::Queue,
    delete_sender: Sender<PainterDelete>,
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        let _ = self
            .delete_sender
            .try_send(PainterDelete::CommandPool(self.command_pool))
            .inspect_err(|e| {
                eprintln!(
                    "error sending drop signal for command pool {:?}: {e}",
                    self.command_pool
                )
            });
    }
}

impl Painter {
    pub fn new_command_pool(&self) -> Result<CommandPool, CommandPoolError> {
        let command_pool = unsafe {
            self.device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(self.graphics_queue_family_index),
                    None,
                )
                .map_err(CommandPoolError::CreateError)?
        };
        Ok(CommandPool {
            command_pool,
            queue: self.graphics_queue,
            delete_sender: self.delete_signal_sender.clone(),
        })
    }

    pub fn allocate_command_buffers(
        &self,
        command_pool: &CommandPool,
        count: usize,
    ) -> Result<Vec<CommandBuffer>, CommandPoolError> {
        unsafe {
            let command_buffers = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(count as u32),
                )
                .map_err(CommandPoolError::CommandBufferAllocationError)?
                .into_iter()
                .map(|command_buffer| CommandBuffer {
                    command_buffer,
                    command_pool: command_pool.command_pool,
                    queue: command_pool.queue,
                })
                .collect();
            Ok(command_buffers)
        }
    }

    pub fn reset_cmd_buffer(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<(), CommandBufferError> {
        unsafe {
            self.device
                .reset_command_buffer(
                    command_buffer.command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .map_err(CommandBufferError::ResetError)
        }
    }

    pub fn record_cmd_buffer(
        &mut self,
        command_buffer: &CommandBuffer,
        commands: &[GpuCommand],
        one_time: bool,
    ) -> Result<(), String> {
        let command_buffer = command_buffer.command_buffer;
        let begin_flags = if one_time {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(begin_flags);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
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
                    self.device.cmd_pipeline_barrier(
                        command_buffer,
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
                    GpuCommand::ImageAccessInit {
                        image: _,
                        access: _,
                    } => {}
                    GpuCommand::ImageAccessHint {
                        image: _,
                        access: _,
                    } => {}
                    GpuCommand::BlitFullImage { src, dst } => {
                        self.device.cmd_blit_image(
                            command_buffer,
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
                        render_output,
                        clear_values,
                        pipelines,
                        pipeline_layouts,
                        commands: rp_commands,
                    } => {
                        self.device.cmd_begin_render_pass(
                            command_buffer,
                            &vk::RenderPassBeginInfo::default()
                                .render_pass(*render_pass)
                                .framebuffer(render_output.framebuffer)
                                .render_area(vk::Rect2D::default().extent(render_output.extent))
                                .clear_values(clear_values),
                            vk::SubpassContents::INLINE,
                        );
                        self.device.cmd_set_viewport(
                            command_buffer,
                            0,
                            &[vk::Viewport::default()
                                .width(render_output.extent.width as f32)
                                .height(render_output.extent.height as f32)],
                        );
                        self.device.cmd_set_scissor(
                            command_buffer,
                            0,
                            &[vk::Rect2D::default().extent(render_output.extent)],
                        );

                        for rp_command in rp_commands.iter() {
                            rp_command.apply_command(
                                &self.device,
                                command_buffer,
                                pipelines,
                                pipeline_layouts
                            );
                        }

                        self.device.cmd_end_render_pass(command_buffer);
                    }
                    GpuCommand::CopyBufferToImageComplete { buffer, image } => {
                        self.device.cmd_copy_buffer_to_image(
                            command_buffer,
                            buffer.buffer,
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

            self.device
                .end_command_buffer(command_buffer)
                .map_err(|e| format!("at command buffer end: {e}"))?;
        }
        Ok(())
    }

    pub fn submit_cmd_buffer(
        &self,
        command_buffer: &CommandBuffer,
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
            self
                .device
                .queue_submit(
                    command_buffer.queue,
                    &[vk::SubmitInfo::default()
                        .signal_semaphores(&signal_semaphores)
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(wait_stages)
                        .command_buffers(&[command_buffer.command_buffer])],
                    vk_fence,
                )
                .map_err(|e| format!("at queue submit: {e}"))?;
        }
        Ok(())
    }
}
