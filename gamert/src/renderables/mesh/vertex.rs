#[derive(Debug, Clone)]
pub struct Vertex {
    pub position: glam::Vec4,
    pub normal: glam::Vec4,
    // tangent: glam::Vec4,
    // bitangent: glam::Vec4,
    pub tex_coords: glam::Vec4,
}
