use std::sync::Arc;

use painter::winit::application::ApplicationHandler;
use painter::winit::event::WindowEvent;
use painter::winit::event_loop::{self, ControlFlow, EventLoop};
use painter::{CommandBuffer, CommandPool, CpuFuture, GpuCommand, GpuFuture, ImageAccess, Painter, Sheets};
use painter::winit::window::{Window, WindowAttributes};

mod mesh_painter;

use mesh_painter::MeshPainter;
use mesh_painter::Vertex;
use mesh_painter::DrawableMeshAndTexture;
use mesh_painter::CamData;

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
    draw_complete_semaphores: Vec<GpuFuture>,
    draw_complete_fences: Vec<CpuFuture>,
    upload_command_buffer: CommandBuffer,
    image_available_semaphore: GpuFuture,
}

impl Canvas {
    pub fn new(window: Window) -> Result<Self, String> {
        let painter = Arc::new(Painter::new(window)?);

        let command_pool = CommandPool::new(painter.clone()).map_err(|e| format!("at create command pool: {e}"))?;

        let upload_command_buffer = command_pool.allocate_command_buffers(1)
            .map_err(|e| format!("at allocate upload command buffer: {e}"))?
            .swap_remove(0);

        let sheets = Sheets::new(painter.clone(), &upload_command_buffer)?;

        let mut mesh_painter = MeshPainter::new(
            painter.clone(),
            sheets.surface_resolution,
            sheets.swapchain_images.len(),
        )?;

        let command_buffers = command_pool.allocate_command_buffers(sheets.swapchain_images.len())
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

        
        let image_available_semaphore = GpuFuture::new(painter.clone())
            .map_err(|e| format!("at create image available semaphore: {e}"))?;

        let square_mesh = mesh_painter.add_mesh(square_verts(), square_indices());
        let default_texture = mesh_painter.add_texture("textures/default.png")
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
            draw_complete_semaphores,
            draw_complete_fences,
            upload_command_buffer,
            image_available_semaphore,
        })
    }

    pub fn paint(&mut self) -> Result<(), String> {
        let frame_num = self.sheets.acquire_next_image(&self.image_available_semaphore, None)
            .expect("Failed to acquire next image");
        let draw_complete_semaphore = &self.draw_complete_semaphores[frame_num as usize];
        let draw_complete_fence = &self.draw_complete_fences[frame_num as usize];
        let cam_data = CamData::new(glam::vec4(0.0, 0.0, -1.0, 1.0), glam::vec4(0.0, 0.0, 0.0, 0.0));

        draw_complete_fence.wait()
            .map_err(|e| format!("at wait for fence: {e}"))?;
        draw_complete_fence.reset()
            .map_err(|e| format!("at reset fence: {e}"))?;

        self.command_buffers[frame_num as usize].reset().map_err(|e| format!("at reset command buffer: {e}"))?;

        self.mesh_painter.update_inputs(
            frame_num as usize, &self.drawables,
            cam_data
        )
            .map_err(|e| format!("at update vb and ib: {e}"))?;
        
        let mesh_render_image = self.mesh_painter.get_rendered_image(frame_num as usize);
        let sheet = &self.sheets.swapchain_images[frame_num as usize];

        let commands = vec![
            GpuCommand::InitialImageAccess { image: mesh_render_image, access: ImageAccess::TransferRead },
            self.mesh_painter.draw_meshes_command(frame_num as usize)
                .map_err(|e| format!("at draw meshes: {e}"))?,
            GpuCommand::InitialImageAccess { image: sheet, access: ImageAccess::TransferWrite },
            GpuCommand::BlitFullImage { src: mesh_render_image, dst: sheet },
            GpuCommand::FinalImageAccess { image: sheet, access: ImageAccess::Present }
        ];
        self.command_buffers[frame_num as usize].record(&commands, false)
            .map_err(|e| format!("at command buffer record: {e}"))?;

        self.command_buffers[frame_num as usize].submit(&[draw_complete_semaphore], &[], &[], Some(draw_complete_fence))
            .map_err(|e| format!("at command buffer submit: {e}"))?;

        self.sheets.present_image(frame_num, &[&self.image_available_semaphore, draw_complete_semaphore]).map_err(|e| format!("at present image: {e}"))?;
    
        Ok(())
    }
}

impl Drop for Canvas {
    fn drop(&mut self) {

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
            return
        }
        let Ok(window) = event_loop
            .create_window(WindowAttributes::default().with_title("Residue2"))
            .inspect_err(|e| eprintln!("at create_window: {e}")) else {
                return
            };
        let Ok(canvas) = Canvas::new(window).inspect_err(|e| eprintln!("at Canvas::new: {e}")) else {
            return
        };
        self.canvas = Some(canvas);
    }

    fn window_event(
        &mut self,
        event_loop: &painter::winit::event_loop::ActiveEventLoop,
        window_id: painter::winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::ActivationTokenDone { serial, token } => {},
            WindowEvent::Resized(physical_size) => {},
            WindowEvent::Moved(physical_position) => {},
            WindowEvent::CloseRequested => {
                event_loop.exit();
            },
            WindowEvent::Destroyed => {},
            WindowEvent::DroppedFile(path_buf) => {},
            WindowEvent::HoveredFile(path_buf) => {},
            WindowEvent::HoveredFileCancelled => {},
            WindowEvent::Focused(_) => {},
            WindowEvent::KeyboardInput { device_id, event, is_synthetic } => {},
            WindowEvent::ModifiersChanged(modifiers) => {},
            WindowEvent::Ime(ime) => {},
            WindowEvent::CursorMoved { device_id, position } => {},
            WindowEvent::CursorEntered { device_id } => {},
            WindowEvent::CursorLeft { device_id } => {},
            WindowEvent::MouseWheel { device_id, delta, phase } => {},
            WindowEvent::MouseInput { device_id, state, button } => {},
            WindowEvent::PinchGesture { device_id, delta, phase } => {},
            WindowEvent::PanGesture { device_id, delta, phase } => {},
            WindowEvent::DoubleTapGesture { device_id } => {},
            WindowEvent::RotationGesture { device_id, delta, phase } => {},
            WindowEvent::TouchpadPressure { device_id, pressure, stage } => {},
            WindowEvent::AxisMotion { device_id, axis, value } => {},
            WindowEvent::Touch(touch) => {},
            WindowEvent::ScaleFactorChanged { scale_factor, inner_size_writer } => {},
            WindowEvent::ThemeChanged(theme) => {},
            WindowEvent::Occluded(_) => {},
            WindowEvent::RedrawRequested => {
                self
                    .canvas
                    .as_mut()
                    .map(|c| c.paint().inspect_err(|e| eprintln!("at paint: {e}")));
            },
        }
    }

    fn about_to_wait(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        self
            .canvas
            .as_mut()
            .map(|c| c.paint().inspect_err(|e| eprintln!("at paint: {e}")));
    }
}

pub fn start_window_event_loop() -> Result<EventLoop<()>, String> {
    let window_event_loop = EventLoop::new().map_err(|e| format!("at EventLoop::new: {e}"))?;
    window_event_loop.set_control_flow(ControlFlow::Poll);
    Ok(window_event_loop)
}
