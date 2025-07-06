use std::sync::Arc;

use painter::winit::application::ApplicationHandler;
use painter::winit::event::WindowEvent;
use painter::winit::event_loop::{self, ControlFlow, EventLoop};
use painter::winit::window::{Window, WindowAttributes};
use painter::{
    CommandBuffer, CommandPool, CpuFuture, GpuCommand, GpuFuture, ImageAccess, Painter, Sheets,
};

mod mesh_painter;

use mesh_painter::CamData;
use mesh_painter::DrawableMeshAndTexture;
use mesh_painter::MeshPainter;
use mesh_painter::Vertex;

fn square_verts() -> Vec<Vertex> {
    vec![
        Vertex {
            position: glam::vec4(-0.5, -0.5, 0.0, 1.0),
            normal: glam::vec4(0.0, 0.0, 1.0, 0.0),
            tex_coords: glam::vec4(0.0, 0.0, 0.0, 0.0),
        },
        Vertex {
            position: glam::vec4(0.5, -0.5, 0.0, 1.0),
            normal: glam::vec4(0.0, 0.0, 1.0, 0.0),
            tex_coords: glam::vec4(1.0, 0.0, 0.0, 0.0),
        },
        Vertex {
            position: glam::vec4(0.5, 0.5, 0.0, 1.0),
            normal: glam::vec4(0.0, 0.0, 1.0, 0.0),
            tex_coords: glam::vec4(1.0, 1.0, 0.0, 0.0),
        },
        Vertex {
            position: glam::vec4(-0.5, 0.5, 0.0, 1.0),
            normal: glam::vec4(0.0, 0.0, 1.0, 0.0),
            tex_coords: glam::vec4(0.0, 1.0, 0.0, 0.0),
        },
    ]
}

fn square_indices() -> Vec<u32> {
    vec![0, 1, 2, 2, 3, 0]
}

pub struct Canvas {
    painter: Arc<Painter>,
    sheets: Sheets,
    mesh_painter: MeshPainter,
    drawables: Vec<DrawableMeshAndTexture>,
    command_pool: CommandPool,
    command_buffers: Vec<CommandBuffer>,
    draw_complete_gpu_futs: Vec<GpuFuture>,
    draw_complete_cpu_futs: Vec<CpuFuture>,
    upload_command_buffer: CommandBuffer,
    acquire_image_cpu_fut: CpuFuture,
}

impl Canvas {
    pub fn new(window: Window) -> Result<Self, String> {
        let painter = Arc::new(Painter::new(window)?);

        let command_pool = CommandPool::new(painter.clone())
            .map_err(|e| format!("at create command pool: {e}"))?;

        let mut upload_command_buffer = command_pool
            .allocate_command_buffers(1)
            .map_err(|e| format!("at allocate upload command buffer: {e}"))?
            .swap_remove(0);

        let sheets = Sheets::new(painter.clone(), &mut upload_command_buffer)?;

        let mut mesh_painter = MeshPainter::new(
            painter.clone(),
            sheets.surface_resolution,
            sheets.swapchain_images.len(),
        )?;

        let command_buffers = command_pool
            .allocate_command_buffers(sheets.swapchain_images.len())
            .map_err(|e| format!("at allocate command buffers: {e}"))?;

        let draw_complete_semaphores = (0..sheets.swapchain_images.len())
            .map(|_| {
                GpuFuture::new(painter.clone())
                    .map_err(|e| format!("at create draw complete semaphore: {e}"))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let draw_complete_fences = (0..sheets.swapchain_images.len())
            .map(|_| {
                CpuFuture::new(painter.clone(), true)
                    .map_err(|e| format!("at create draw complete fence: {e}"))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let acquire_image_future = CpuFuture::new(painter.clone(), false)
            .map_err(|e| format!("at create acquire image future: {e}"))?;

        let square_mesh = mesh_painter.add_mesh(square_verts(), square_indices());
        let default_texture = mesh_painter
            .add_texture("textures/default.png")
            .map_err(|e| format!("at add default texture: {e}"))?;
        Ok(Self {
            painter,
            sheets,
            mesh_painter,
            drawables: vec![DrawableMeshAndTexture {
                mesh_name: square_mesh,
                texture_name: default_texture,
            }],
            command_pool,
            command_buffers,
            draw_complete_gpu_futs: draw_complete_semaphores,
            draw_complete_cpu_futs: draw_complete_fences,
            upload_command_buffer,
            acquire_image_cpu_fut: acquire_image_future,
        })
    }

    pub fn paint(&mut self) -> Result<(), String> {
        // Wait till next image is available
        let frame_num = self
            .sheets
            .acquire_next_image(None, Some(&self.acquire_image_cpu_fut), &mut self.upload_command_buffer)
            .map_err(|e| format!("at acquire next image: {e}"))?;
        self.acquire_image_cpu_fut
            .wait()
            .map_err(|e| format!("at wait for acquire image future: {e}"))?;
        self.acquire_image_cpu_fut
            .reset()
            .map_err(|e| format!("at reset acquire image future: {e}"))?;

        let draw_complete_gpu_fut = &self.draw_complete_gpu_futs[frame_num as usize];
        let draw_complete_cpu_fut = &self.draw_complete_cpu_futs[frame_num as usize];

        draw_complete_cpu_fut
            .wait()
            .map_err(|e| format!("at wait for draw complete cpu future: {e}"))?;
        draw_complete_cpu_fut
            .reset()
            .map_err(|e| format!("at reset draw complete cpu future: {e}"))?;

        let cam_data = CamData::new(
            glam::vec4(0.0, 0.0, 1.0, 1.0),
            glam::vec4(0.0, 0.0, 0.0, 0.0),
        );

        self.command_buffers[frame_num as usize]
            .reset()
            .map_err(|e| format!("at reset command buffer: {e}"))?;

        self.mesh_painter
            .update_inputs(frame_num as usize, &self.drawables, cam_data)
            .map_err(|e| format!("at update vb and ib: {e}"))?;

        let mesh_render_image = self.mesh_painter.get_rendered_image(frame_num as usize);
        let sheet = &self.sheets.swapchain_images[frame_num as usize];

        let commands = vec![
            GpuCommand::ImageAccessHint {
                image: mesh_render_image,
                access: ImageAccess::TransferRead,
            },
            GpuCommand::ImageAccessHint {
                image: mesh_render_image,
                access: ImageAccess::PipelineAttachment,
            },
            self.mesh_painter
                .draw_meshes_command(frame_num as usize)
                .map_err(|e| format!("at draw meshes: {e}"))?,
            GpuCommand::ImageAccessHint {
                image: sheet,
                access: ImageAccess::Present,
            },
            GpuCommand::BlitFullImage {
                src: mesh_render_image,
                dst: sheet,
            },
            GpuCommand::ImageAccessHint {
                image: sheet,
                access: ImageAccess::Present,
            },
        ];
        self.command_buffers[frame_num as usize]
            .record(&commands, false)
            .map_err(|e| format!("at command buffer record: {e}"))?;

        self.command_buffers[frame_num as usize]
            .submit(
                &[draw_complete_gpu_fut],
                &[],
                &[],
                Some(&draw_complete_cpu_fut),
            )
            .map_err(|e| format!("at command buffer submit: {e}"))?;

        self.sheets
            .present_image(frame_num, &[draw_complete_gpu_fut])
            .map_err(|e| format!("at present image: {e}"))?;
        Ok(())
    }
}

impl Drop for Canvas {
    fn drop(&mut self) {
        unsafe {
            self.painter
                .device
                .device_wait_idle()
                .map_err(|e| eprintln!("at wait for device idle: {e}"))
                .ok();
        }
    }
}

pub struct Game {
    canvas: Option<Canvas>,
}

impl Game {
    pub fn new() -> Self {
        Self { canvas: None }
    }
}

impl ApplicationHandler for Game {
    fn resumed(&mut self, event_loop: &painter::winit::event_loop::ActiveEventLoop) {
        if self.canvas.is_some() {
            return;
        }
        let Ok(window) = event_loop
            .create_window(WindowAttributes::default().with_title("Residue2"))
            .inspect_err(|e| eprintln!("at create_window: {e}"))
        else {
            return;
        };
        let Ok(canvas) = Canvas::new(window).inspect_err(|e| eprintln!("at Canvas::new: {e}"))
        else {
            return;
        };
        self.canvas = Some(canvas);
    }

    fn window_event(
        &mut self,
        event_loop: &painter::winit::event_loop::ActiveEventLoop,
        _window_id: painter::winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::ActivationTokenDone { serial: _, token: _ } => {}
            WindowEvent::Resized(_physical_size) => {}
            WindowEvent::Moved(_physical_position) => {}
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Destroyed => {}
            WindowEvent::DroppedFile(_path_buf) => {}
            WindowEvent::HoveredFile(_path_buf) => {}
            WindowEvent::HoveredFileCancelled => {}
            WindowEvent::Focused(_) => {}
            WindowEvent::KeyboardInput {
                device_id: _,
                event: _,
                is_synthetic: _,
            } => {}
            WindowEvent::ModifiersChanged(_modifiers) => {}
            WindowEvent::Ime(_ime) => {}
            WindowEvent::CursorMoved {
                device_id: _,
                position: _,
            } => {}
            WindowEvent::CursorEntered { device_id: _ } => {}
            WindowEvent::CursorLeft { device_id: _ } => {}
            WindowEvent::MouseWheel {
                device_id: _,
                delta: _,
                phase: _,
            } => {}
            WindowEvent::MouseInput {
                device_id: _,
                state: _,
                button: _,
            } => {}
            WindowEvent::PinchGesture {
                device_id: _,
                delta: _,
                phase: _,
            } => {}
            WindowEvent::PanGesture {
                device_id: _,
                delta: _,
                phase: _,
            } => {}
            WindowEvent::DoubleTapGesture { device_id: _ } => {}
            WindowEvent::RotationGesture {
                device_id: _,
                delta: _,
                phase: _,
            } => {}
            WindowEvent::TouchpadPressure {
                device_id: _,
                pressure: _,
                stage: _,
            } => {}
            WindowEvent::AxisMotion {
                device_id: _,
                axis: _,
                value: _,
            } => {}
            WindowEvent::Touch(_touch) => {}
            WindowEvent::ScaleFactorChanged {
                scale_factor: _,
                inner_size_writer: _,
            } => {}
            WindowEvent::ThemeChanged(_theme) => {}
            WindowEvent::Occluded(_) => {}
            WindowEvent::RedrawRequested => {
                self.canvas
                    .as_mut()
                    .map(|c| c.paint().inspect_err(|e| eprintln!("at paint: {e}")));
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &event_loop::ActiveEventLoop) {
        self.canvas
            .as_mut()
            .map(|c| c.paint().inspect_err(|e| eprintln!("at paint: {e}")));
    }
}

pub fn start_window_event_loop() -> Result<EventLoop<()>, String> {
    let window_event_loop = EventLoop::new().map_err(|e| format!("at EventLoop::new: {e}"))?;
    window_event_loop.set_control_flow(ControlFlow::Poll);
    Ok(window_event_loop)
}
