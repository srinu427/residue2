mod vertex;

pub use vertex::Vertex;

#[derive(Debug, Clone)]
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}
