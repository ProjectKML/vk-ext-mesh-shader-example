use std::{mem, path::Path, slice, sync::Arc};

use anyhow::Result;
use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use meshopt::VertexDataAdapter;
use vk_mem_alloc::Allocator;

use crate::{render::buffer::Buffer, RenderCtx};

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub tex_coord: Vec2,
    pub normal: Vec3
}

unsafe impl Zeroable for Vertex {}
unsafe impl Pod for Vertex {}

impl Vertex {
    #[inline]
    pub fn new(position: Vec3, tex_coord: Vec2, normal: Vec3) -> Self {
        Self { position, tex_coord, normal }
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Meshlet {
    pub data_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32
}

unsafe impl Zeroable for Meshlet {}
unsafe impl Pod for Meshlet {}

impl Meshlet {
    #[inline]
    pub fn new(data_offset: u32, vertex_count: u32, triangle_count: u32) -> Self {
        Self {
            data_offset,
            vertex_count,
            triangle_count
        }
    }
}

const MAX_VERTICES: usize = 64;
const MAX_TRIANGLES: usize = 124;
const CONE_WEIGHT: f32 = 0.0;

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub meshlets: Vec<Meshlet>,
    pub meshlet_data: Vec<u32>
}

impl Mesh {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let mesh = fast_obj::Mesh::new(path)?;

        let mut vertices = vec![Default::default(); mesh.indices().len()];

        let positions = mesh.positions();
        let tex_coords = mesh.texcoords();
        let normals = mesh.normals();
        let indices = mesh.indices();

        for (i, index) in indices.iter().enumerate() {
            let position_idx = 3 * index.p as usize;
            let tex_coord_idx = 2 * index.t as usize;
            let normal_idx = 3 * index.n as usize;

            vertices[i] = Vertex::new(
                Vec3::new(positions[position_idx], positions[position_idx + 1], positions[position_idx + 2]),
                Vec2::new(tex_coords[tex_coord_idx], tex_coords[tex_coord_idx + 1]),
                Vec3::new(normals[normal_idx], normals[normal_idx + 1], normals[normal_idx + 2])
            );
        }

        let (vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, None);
        vertices.shrink_to(vertex_count);

        let mut vertices = meshopt::remap_vertex_buffer(&vertices, vertex_count, &remap);
        let mut indices = meshopt::remap_index_buffer(None, indices.len(), &remap);

        meshopt::optimize_vertex_cache_in_place(&mut indices, vertices.len());
        meshopt::optimize_overdraw_in_place(&mut indices, &VertexDataAdapter::new(bytemuck::cast_slice(&vertices), mem::size_of::<Vertex>(), 0)?, 1.01);
        meshopt::optimize_vertex_fetch_in_place(&mut indices, &mut vertices);

        let meshlets = meshopt::build_meshlets(
            &indices,
            &VertexDataAdapter::new(bytemuck::cast_slice(&vertices), mem::size_of::<Vertex>(), 0)?,
            MAX_VERTICES,
            MAX_TRIANGLES,
            CONE_WEIGHT
        );

        let num_meshlet_data = meshlets.iter().map(|meshlet| meshlet.vertices.len() + ((meshlet.triangles.len() * 3 + 3) >> 2)).sum();

        let mut meshlet_data = vec![0; num_meshlet_data];

        let mut index = 0;
        let meshlets = meshlets
            .iter()
            .map(|meshlet| {
                let data_offset = index;

                for vertex in meshlet.vertices {
                    meshlet_data[index] = *vertex;
                    index += 1;
                }

                let num_packed_indices = (meshlet.triangles.len() + 3) >> 2;
                for j in 0..num_packed_indices {
                    let triangle_offset = j << 2;
                    meshlet_data[index] = (meshlet.triangles[triangle_offset] as u32) << 24
                        | (meshlet.triangles.get(triangle_offset + 1).copied().unwrap_or_default() as u32) << 16
                        | (meshlet.triangles.get(triangle_offset + 2).copied().unwrap_or_default() as u32) << 8
                        | (meshlet.triangles.get(triangle_offset + 3).copied().unwrap_or_default() as u32);
                    index += 1;
                }

                Meshlet::new(data_offset as _, meshlet.vertices.len() as _, (meshlet.triangles.len() / 3) as _)
            })
            .collect();

        Ok(Self { vertices, meshlets, meshlet_data })
    }
}

#[derive(Clone)]
pub struct MeshBuffers {
    pub vertex_buffer: Buffer,
    pub meshlet_buffer: Buffer,
    pub meshlet_data_buffer: Buffer,
    pub num_meshlets: usize
}

impl MeshBuffers {
    pub unsafe fn new(device: Arc<Device>, queue: vk::Queue, allocator: Allocator, path: impl AsRef<Path>) -> Result<Self> {
        let mesh = Mesh::new(path)?;

        let vertex_buffer = Buffer::new_device_local(device.clone(), queue, allocator, &mesh.vertices)?;
        let meshlet_buffer = Buffer::new_device_local(device.clone(), queue, allocator, &mesh.meshlets)?;
        let meshlet_data_buffer = Buffer::new_device_local(device.clone(), queue, allocator, &mesh.meshlet_data)?;

        Ok(Self {
            vertex_buffer,
            meshlet_buffer,
            meshlet_data_buffer,
            num_meshlets: mesh.meshlets.len()
        })
    }
}

#[derive(Clone)]
pub struct MeshCollection {
    mesh_buffers: Vec<MeshBuffers>,
    _address_buffer: Buffer,
    descriptor_set: vk::DescriptorSet
}

impl MeshCollection {
    pub unsafe fn new<P: AsRef<Path>>(
        device: &Arc<Device>,
        queue: vk::Queue,
        allocator: Allocator,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
        names: impl IntoIterator<Item = P>
    ) -> Result<Self> {
        let mesh_buffers = names
            .into_iter()
            .map(|name| MeshBuffers::new(device.clone(), queue, allocator, name))
            .collect::<Result<Vec<_>, _>>()?;

        let addresses: Vec<_> = mesh_buffers
            .iter()
            .flat_map(|mesh_buffers| {
                [
                    mesh_buffers.vertex_buffer.device_address,
                    mesh_buffers.meshlet_buffer.device_address,
                    mesh_buffers.meshlet_data_buffer.device_address
                ]
            })
            .collect();

        let address_buffer = Buffer::new_device_local(device.clone(), queue, allocator, &addresses)?;

        let descriptor_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(slice::from_ref(&descriptor_set_layout))
        )?[0];

        let descriptor_buffer_info = vk::DescriptorBufferInfo::default().buffer(address_buffer.buffer).range(address_buffer.size);

        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(slice::from_ref(&descriptor_buffer_info));

        device.update_descriptor_sets(slice::from_ref(&write_descriptor_set), &[]);

        Ok(Self {
            mesh_buffers,
            _address_buffer: address_buffer,
            descriptor_set
        })
    }

    pub unsafe fn bind(&self, ctx: &RenderCtx, command_buffer: vk::CommandBuffer) {
        ctx.device_loader.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            ctx.pipeline_layout,
            0,
            slice::from_ref(&self.descriptor_set),
            &[]
        );
    }

    pub unsafe fn draw_mesh(&self, ctx: &RenderCtx, command_buffer: vk::CommandBuffer, mvp_matrix: &Mat4, mesh_idx: u32) {
        ctx.device_loader.cmd_push_constants(
            command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::MESH_EXT,
            0,
            slice::from_raw_parts(mvp_matrix as *const Mat4 as *const _, mem::size_of::<Mat4>())
        );

        ctx.device_loader.cmd_push_constants(
            command_buffer,
            ctx.pipeline_layout,
            vk::ShaderStageFlags::MESH_EXT,
            mem::size_of::<Mat4>() as _,
            slice::from_raw_parts(&mesh_idx as *const u32 as *const _, mem::size_of::<u32>())
        );

        let num_meshlets = self.mesh_buffers[mesh_idx as usize].num_meshlets;

        ctx.mesh_shader_loader.cmd_draw_mesh_tasks(command_buffer, ((num_meshlets * 32 + 31) >> 5) as u32, 1, 1)
    }
}
