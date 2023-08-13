use glam::Vec3;

use crate::render::mesh::Vertex;

#[derive(Clone, Debug)]
pub struct MeshBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn add_quad(&mut self, v0: &Vertex, v1: &Vertex, v2: &Vertex, v3: &Vertex) {
        let idx_offset = self.vertices.len() as u32;

        self.vertices.push(*v0);
        self.vertices.push(*v1);
        self.vertices.push(*v2);
        self.vertices.push(*v3);

        self.indices.push(idx_offset);
        self.indices.push(idx_offset + 1);
        self.indices.push(idx_offset + 3);
        self.indices.push(idx_offset + 3);
        self.indices.push(idx_offset + 1);
        self.indices.push(idx_offset + 2);
    }

    pub fn add_nano_mesh(&mut self, pos: &Vec3, scale: f32) {
        self.add_quad(
            &Vertex {
                position: *pos + Vec3::new(-0.002, 0.0, 0.0) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.002, 0.0, 0.0) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.002, 0.01, 0.0) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(-0.002, 0.01, 0.0) * scale,
                ..Default::default()
            },
        );

        self.add_quad(
            &Vertex {
                position: *pos + Vec3::new(0.0, 0.0, -0.002) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.0, 0.0, 0.002) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.0, 0.01, 0.002) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.0, 0.01, -0.002) * scale,
                ..Default::default()
            },
        );

        self.add_quad(
            &Vertex {
                position: *pos + Vec3::new(-0.002, 0.01, -0.002) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.002, 0.01, -0.002) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(0.002, 0.01, 0.002) * scale,
                ..Default::default()
            },
            &Vertex {
                position: *pos + Vec3::new(-0.002, 0.01, 0.002) * scale,
                ..Default::default()
            },
        );
    }

    pub fn build(mut self) -> (Vec<Vertex>, Vec<u32>) {
        (self.vertices, self.indices)
    }
}
