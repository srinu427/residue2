use std::{collections::HashMap, mem::offset_of, sync::Arc};

use ash::vk;
use glam::Vec4Swizzles;
use include_bytes_aligned::include_bytes_aligned;
use painter::{
    ash, slotmap::{new_key_type, SlotMap}, GAllocator, Buffer, CommandBuffer, CommandPool, CpuFuture, GpuCommand, GpuRenderPassCommand, Image2d, ImageAccess, Painter, RenderOutput, ShaderInputAllocator, ShaderInputBindingInfo, ShaderInputType, SingePassRenderPipeline
};

static VERTEX_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/mesh_painter.vert.spv");
static FRAGMENT_SHADER_CODE: &[u8] = include_bytes_aligned!(4, "shaders/mesh_painter.frag.spv");

static MAX_TEXTURES: usize = 100;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CamData {
    pub pos: glam::Vec4,
    pub look_at: glam::Vec4,
    pub view_proj: glam::Mat4,
}

impl CamData {
    pub fn new(pos: glam::Vec4, look_at: glam::Vec4) -> Self {
        let view = glam::Mat4::look_at_rh(pos.xyz(), look_at.xyz(), glam::Vec3::new(0.0, 1.0, 0.0));
        let proj = glam::Mat4::perspective_rh(90.0f32.to_radians(), 1.0, 0.1, 100.0);
        let view_proj = proj * view;
        Self {
            pos,
            look_at,
            view_proj,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct SceneDescriptorData {
    cam_data: CamData,
}

pub struct PerFrameData {
    descriptor_sets: Vec<vk::DescriptorSet>,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_buffer_size: u32,
    scene_buffer: Buffer,
    color_image: Image2d,
    depth_image: Image2d,
    render_output: RenderOutput,
}

impl PerFrameData {
    pub fn new(
        pipeline: &SingePassRenderPipeline,
        allocator: &mut GAllocator,
        color_format: vk::Format,
        depth_format: vk::Format,
        extent: vk::Extent2D,
        shader_input_allocator: &ShaderInputAllocator,
        command_buffer: &mut CommandBuffer,
    ) -> Result<Self, String> {
        let descriptor_sets = pipeline
            .make_shader_inputs(shader_input_allocator)
            .map_err(|e| format!("at make shader inputs: {e}"))?;
        let painter = pipeline.painter.clone();
        let vertex_buffer = Buffer::new_with_mem(
            painter.clone(),
            32 * 1024 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            allocator,
            false
        )
            .map_err(|e| format!("at create vertex buffer: {e}"))?;

        let index_buffer = Buffer::new_with_mem(
            painter.clone(),
            4 * 1024 * 1024,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            allocator,
            false
        )
            .map_err(|e| format!("at create vertex buffer: {e}"))?;

        let scene_buffer = Buffer::new_with_mem(
            painter.clone(),
            size_of::<SceneDescriptorData>() as _,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            allocator,
            false
        )
            .map_err(|e| format!("at create vertex buffer: {e}"))?;

        let color_image = Image2d::new_with_mem(
            painter.clone(),
            color_format,
            extent,
            vec![ImageAccess::PipelineAttachment, ImageAccess::TransferRead],
            allocator,
            true
        )
            .map_err(|e| format!("at create color image: {e}"))?;

        let depth_image = Image2d::new_with_mem(
            painter.clone(),
            depth_format,
            extent,
            vec![ImageAccess::PipelineAttachment],
            allocator,
            true
        )
            .map_err(|e| format!("at create depth image: {e}"))?;

        let commands = vec![
            GpuCommand::ImageAccessInit {
                image: &color_image,
                access: ImageAccess::TransferRead,
            },
            GpuCommand::ImageAccessInit {
                image: &depth_image,
                access: ImageAccess::PipelineAttachment,
            },
        ];

        command_buffer
            .record(&commands, true)
            .map_err(|e| format!("at record command buffer: {e}"))?;

        let fence = CpuFuture::new(pipeline.painter.clone(), false)
            .map_err(|e| format!("at create fence: {e}"))?;
        command_buffer
            .submit(&[], &[], &[], Some(&fence))
            .map_err(|e| format!("at submit command buffer: {e}"))?;
        fence.wait().map_err(|e| format!("at fence wait: {e}"))?;
        command_buffer
            .reset()
            .map_err(|e| format!("at reset command buffer: {e}"))?;

        let render_output = pipeline
            .create_render_output(vec![&color_image, &depth_image])
            .map_err(|e| format!("at create render output: {e}"))?;

        Ok(Self {
            descriptor_sets,
            vertex_buffer,
            index_buffer,
            scene_buffer,
            index_buffer_size: 0,
            color_image,
            depth_image,
            render_output,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplingMode {
    X1 = 0,
    X4 = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct DrawableMeshAndTexture {
    pub mesh_name: MeshID,
    pub texture_name: TextureID,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuObjectInfo {
    pub obj_id: u32,
    pub mesh_id: u32,
    pub texture_id: u32,
}

#[derive(Debug, Clone)]
pub struct ObjDrawParams {
    pub vert_offset: i32,
    pub idx_offset: u32,
    pub idx_count: u32,
    pub obj_info: GpuObjectInfo,
}

#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: glam::Vec4,
    pub normal: glam::Vec4,
    // tangent: glam::Vec4,
    // bitangent: glam::Vec4,
    pub tex_coords: glam::Vec4,
}

impl Vertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription> {
        vec![
            vk::VertexInputBindingDescription::default()
                .stride(size_of::<Self>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
        ]
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .offset(offset_of!(Self, position) as u32)
                .format(vk::Format::R32G32B32A32_SFLOAT),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .offset(offset_of!(Self, normal) as u32)
                .format(vk::Format::R32G32B32A32_SFLOAT),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .offset(offset_of!(Self, tex_coords) as u32)
                .format(vk::Format::R32G32B32A32_SFLOAT),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

new_key_type! {
    pub struct MeshID;
}

new_key_type! {
    pub struct TextureID;
}

pub struct MeshPainter {
    painter: Arc<Painter>,
    pipeline: SingePassRenderPipeline,
    color_attachment_format: vk::Format,
    depth_attachment_format: vk::Format,
    sampler: vk::Sampler,
    allocator: GAllocator,
    meshes: SlotMap<MeshID, Mesh>,
    textures: SlotMap<TextureID, Image2d>,
    textures_to_delete: Vec<Image2d>,
    shader_input_allocator: ShaderInputAllocator,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    per_frame_datas: Vec<PerFrameData>,
}

impl MeshPainter {
    fn select_depth_format(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::Format, String> {
        let preferred_depth_formats = [
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
        ];
        for &format in &preferred_depth_formats {
            let properties =
                unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            if properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                return Ok(format);
            }
        }
        return Err("No suitable depth format found".to_string());
    }

    pub fn new(
        painter: Arc<Painter>,
        resolution: vk::Extent2D,
        frame_count: usize,
    ) -> Result<Self, String> {
        unsafe {
            let device = &painter.device;

            let color_attachment_format = vk::Format::R8G8B8A8_UNORM;
            let depth_attachment_format =
                Self::select_depth_format(&painter.instance, painter.physical_device)
                    .map_err(|e| format!("at select depth format: {e}"))?;

            let sampler = device
                .create_sampler(&vk::SamplerCreateInfo::default(), None)
                .map_err(|e| format!("at create sampler: {e}"))?;

            let pipeline = SingePassRenderPipeline::new(
                painter.clone(),
                vec![(
                    color_attachment_format,
                    vk::AttachmentLoadOp::CLEAR,
                    vk::AttachmentStoreOp::STORE,
                )],
                Some((
                    depth_attachment_format,
                    vk::AttachmentLoadOp::CLEAR,
                    vk::AttachmentStoreOp::DONT_CARE,
                )),
                vec![
                    vec![
                        ShaderInputBindingInfo {
                            _type: ShaderInputType::StorageBuffer,
                            count: 1,
                            dynamic: false,
                        },
                        ShaderInputBindingInfo {
                            _type: ShaderInputType::Sampler,
                            count: 1,
                            dynamic: false,
                        },],
                    vec![
                        
                        ShaderInputBindingInfo {
                            _type: ShaderInputType::SampledImage2d,
                            count: MAX_TEXTURES as _,
                            dynamic: true,
                        },
                    ],
                ],
                size_of::<GpuObjectInfo>(),
                VERTEX_SHADER_CODE,
                FRAGMENT_SHADER_CODE,
                Vertex::get_binding_description(),
                Vertex::get_attribute_descriptions(),
            )
            .map_err(|e| format!("at create render pipeline: {e}"))?;

            let shader_input_allocator = ShaderInputAllocator::new(
                painter.clone(),
                vec![
                    (ShaderInputType::StorageBuffer, frame_count as u32),
                    (ShaderInputType::Sampler, 2),
                    (
                        ShaderInputType::SampledImage2d,
                        (MAX_TEXTURES * frame_count) as u32,
                    ),
                ],
                4 * frame_count as u32,
            )
            .map_err(|e| format!("at create shader input allocator: {e}"))?;

            let mut allocator =
                GAllocator::new(painter.clone()).map_err(|e| format!("at create allocator: {e}"))?;

            let command_pool = CommandPool::new(painter.clone())
                .map_err(|e| format!("at create command pool: {e}"))?;

            let mut command_buffer = command_pool
                .allocate_command_buffers(1)
                .map_err(|e| format!("at allocate command buffer: {e}"))?
                .swap_remove(0);

            let per_frame_datas = (0..frame_count)
                .map(|_| {
                    PerFrameData::new(
                        &pipeline,
                        &mut allocator,
                        color_attachment_format,
                        depth_attachment_format,
                        resolution,
                        &shader_input_allocator,
                        &mut command_buffer,
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok(Self {
                painter,
                pipeline,
                color_attachment_format,
                depth_attachment_format,
                meshes: SlotMap::with_key(),
                textures: SlotMap::with_key(),
                textures_to_delete: Vec::new(),
                shader_input_allocator,
                command_pool,
                command_buffer,
                per_frame_datas,
                sampler,
                allocator,
            })
        }
    }

    pub fn get_rendered_image(&self, frame_number: usize) -> &Image2d {
        &self.per_frame_datas[frame_number % self.per_frame_datas.len()].color_image
    }

    pub fn add_mesh(&mut self, vertices: Vec<Vertex>, indices: Vec<u32>) -> MeshID {
        let mesh_id = self.meshes.insert(Mesh { vertices, indices });
        mesh_id
    }

    pub fn add_texture(&mut self, path: &str) -> Result<TextureID, String> {
        let image = image::open(path).map_err(|e| format!("at open image: {e}"))?;
        let image_data = image.to_rgba8();
        let vk_image = Image2d::new_with_mem(
            self.painter.clone(),
            vk::Format::R8G8B8A8_UNORM,
            vk::Extent2D {
                width: image.width(),
                height: image.height(),
            },
            vec![ImageAccess::TransferWrite, ImageAccess::ShaderRead],
            &mut self.allocator,
            true
        )
            .map_err(|e| format!("at vk create image: {e}"))?;

        let stage_buffer = Buffer::new_with_mem(self.painter.clone(), image_data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, &mut self.allocator, false).map_err(|e| format!("at create stage buffer: {e}"))?;

        self.allocator
            .write_to_mem(stage_buffer.get_allocation_id().ok_or("mem not allocated???".to_string())?, &image_data)
            .map_err(|e| format!("at write to staging buffer mem: {e}"))?;

        let commands = vec![
            GpuCommand::ImageAccessInit {
                image: &vk_image,
                access: ImageAccess::TransferWrite,
            },
            GpuCommand::CopyBufferToImageComplete {
                buffer: &stage_buffer,
                image: &vk_image,
            },
            GpuCommand::ImageAccessHint {
                image: &vk_image,
                access: ImageAccess::ShaderRead,
            },
        ];
        self.command_buffer
            .record(&commands, true)
            .map_err(|e| format!("at record command buffer: {e}"))?;

        let fence = CpuFuture::new(self.painter.clone(), false)
            .map_err(|e| format!("at create upload texture fence: {e}"))?;
        self.command_buffer
            .submit(&[], &[], &[], Some(&fence))
            .map_err(|e| format!("at submit command buffer: {e}"))?;
        fence
            .wait()
            .map_err(|e| format!("at texture upload fence wait: {e}"))?;

        let texture_id = self.textures.insert(vk_image);
        Ok(texture_id)
    }

    pub fn update_inputs(
        &mut self,
        frame_number: usize,
        drawables: &[DrawableMeshAndTexture],
        camera: CamData,
    ) -> Result<(), String> {
        let mut vb_data = vec![];
        let mut ib_data = vec![];

        let mut vb_offset = 0i32;
        let mut ib_offset = 0;
        let mut mesh_id = 0;

        let textures_array = self.textures.iter().collect::<Vec<_>>();

        let texture_idx_map = textures_array
            .iter()
            .enumerate()
            .map(|(tid, tex)| (tex.0, tid))
            .collect::<HashMap<_, _>>();

        let mut objects = vec![];

        for drawable in drawables {
            let Some(mesh) = self.meshes.get(drawable.mesh_name) else {
                continue;
            };
            let Some(&texture_idx) = texture_idx_map.get(&drawable.texture_name) else {
                continue;
            };
            vb_data.extend_from_slice(&mesh.vertices);
            ib_data.extend_from_slice(
                &mesh
                    .indices
                    .iter()
                    .map(|i| i + vb_offset as u32)
                    .collect::<Vec<_>>(),
            );

            let object = GpuObjectInfo {
                obj_id: objects.len() as u32,
                mesh_id,
                texture_id: texture_idx as u32,
            };
            mesh_id += 1;
            objects.push(ObjDrawParams {
                vert_offset: vb_offset,
                idx_offset: ib_offset,
                idx_count: mesh.indices.len() as u32,
                obj_info: object,
            });
            vb_offset += mesh.vertices.len() as i32;
            ib_offset += mesh.indices.len() as u32;
        }

        let norm_frame_number = frame_number % self.per_frame_datas.len();
        self.per_frame_datas[norm_frame_number].index_buffer_size = ib_data.len() as u32;
        self.per_frame_datas[norm_frame_number].next_draw_params = objects;

        let vb = &self.per_frame_datas[norm_frame_number].vertex_buffer;
        let ib = &self.per_frame_datas[norm_frame_number].index_buffer;
        let sb = &self.per_frame_datas[norm_frame_number].scene_buffer;

        unsafe {
            let scene_data = SceneDescriptorData { cam_data: camera };
            self.allocator
                .write_to_mem(sb.get_allocation_id().ok_or("mem not allocated???".to_string())?, &[scene_data].align_to::<u8>().1)
                .map_err(|e| format!("at write to scene buffer mem: {e}"))?;
            self.allocator
                .write_to_mem(vb.get_allocation_id().ok_or("mem not allocated???".to_string())?, vb_data.as_slice().align_to::<u8>().1)
                .map_err(|e| format!("at write to vertex buffer mem: {e}"))?;
            self.allocator
                .write_to_mem(ib.get_allocation_id().ok_or("mem not allocated???".to_string())?, ib_data.as_slice().align_to::<u8>().1)
                .map_err(|e| format!("at write to index buffer mem: {e}"))?;

            let scene_dset = self.per_frame_datas[norm_frame_number].descriptor_sets[0];
            let texture_dset = self.per_frame_datas[norm_frame_number].descriptor_sets[1];

            self.painter.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(scene_dset)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(sb.buffer)
                            .range(vk::WHOLE_SIZE)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(scene_dset)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .descriptor_count(1)
                        .image_info(&[vk::DescriptorImageInfo::default().sampler(self.sampler)]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(texture_dset)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(textures_array.len() as _)
                        .image_info(
                            &textures_array
                                .iter()
                                .map(|(_, tex)| {
                                    vk::DescriptorImageInfo::default()
                                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                        .image_view(tex.image_view)
                                })
                                .collect::<Vec<_>>(),
                        ),
                ],
                &[],
            );
            // println!("number of textures written: {}", textures_array.len());
        }

        Ok(())
    }

    pub fn draw_meshes_command(&self, frame_number: usize) -> Result<GpuCommand, String> {
        let frame_number = frame_number % self.per_frame_datas.len();
        let per_frame_data = &self.per_frame_datas[frame_number];
        let mut render_cmds = vec![];
        render_cmds.push(GpuRenderPassCommand::BindPipeline { pipeline: 0 });
        render_cmds.push(GpuRenderPassCommand::BindVertexBuffers {
            buffers: vec![&per_frame_data.vertex_buffer],
        });
        render_cmds.push(GpuRenderPassCommand::BindIndexBuffer {
            buffer: &per_frame_data.index_buffer,
        });
        render_cmds.push(GpuRenderPassCommand::BindShaderInput {
            pipeline_layout: 0,
            descriptor_sets: per_frame_data.descriptor_sets.clone(),
        });
        for draw_param in &per_frame_data.next_draw_params {
            unsafe {
                render_cmds.push(GpuRenderPassCommand::SetPushConstant {
                    pipeline_layout: 0,
                    data: [draw_param.obj_info].align_to::<u8>().1.to_vec(),
                });
            }
            render_cmds.push(GpuRenderPassCommand::Draw {
                count: draw_param.idx_count,
                vertex_offset: draw_param.vert_offset,
                index_offset: draw_param.idx_offset,
            });
        }
        let gpu_command = GpuCommand::RunRenderPass {
            render_pass: self.pipeline.render_pass,
            render_output: &per_frame_data.render_output,
            clear_values: vec![
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [1.0, 1.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ],
            pipelines: vec![self.pipeline.pipeline],
            pipeline_layouts: vec![self.pipeline.pipeline_layout],
            commands: render_cmds,
        };
        Ok(gpu_command)
    }
}

impl Drop for MeshPainter {
    fn drop(&mut self) {
        let device = &self.painter.device;
        self.textures_to_delete.clear();
        self.textures.clear();
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
    }
}
