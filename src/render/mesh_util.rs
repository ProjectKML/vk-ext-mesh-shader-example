use glam::Vec3;

use crate::render::mesh::Vertex;

#[derive(Copy, Clone, Debug, Default)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn from_vertices<'a>(vertices: impl Iterator<Item = &'a Vertex>) -> Self {
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        vertices.for_each(|v| {
            min.x = min.x.min(v.position.x);
            min.y = min.y.min(v.position.y);
            min.z = min.z.min(v.position.z);

            max.x = max.x.max(v.position.x);
            max.y = max.y.max(v.position.y);
            max.z = max.z.max(v.position.z);
        });

        Self { min, max }
    }

    #[inline]
    pub fn range(&self) -> f32 {
        (self.max.x - self.min.x)
            .max(self.max.y - self.min.y)
            .max(self.max.z - self.min.z)
    }
}
