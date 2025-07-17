use glam::Vec4Swizzles;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Camera {
    pub pos: glam::Vec4,
    pub look_at: glam::Vec4,
    pub view_proj: glam::Mat4,
}

impl Camera {
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
