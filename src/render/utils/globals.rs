use std::{mem, slice, sync::Arc};

use ash::{vk, Device};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use vk_mem_alloc::Allocator;

use crate::render::buffer::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Zeroable, Pod)]
pub struct Globals {
    pub view_projection_matrix: Mat4,
    pub frustum_planes: [Vec4; 6],
    pub camera_pos: Vec3,
    pub time: f32,
}

pub struct GlobalsBuffers {
    pub uniform_buffer: Buffer,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    device: Arc<Device>,
}

impl Drop for GlobalsBuffers {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl GlobalsBuffers {
    pub fn new(
        device: &Arc<Device>,
        allocator: Allocator,
        descriptor_pool: vk::DescriptorPool,
    ) -> Self {
        //Create uniform buffer
        let uniform_buffer =
            unsafe { Buffer::new_uniform(device.clone(), allocator, mem::size_of::<Globals>()) }
                .unwrap();

        //Create descriptor set layout
        let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::COMPUTE);

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(slice::from_ref(&descriptor_set_layout_binding));

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }
        .unwrap();

        //Create descriptor set
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(slice::from_ref(&descriptor_set_layout));

        let descriptor_set =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }.unwrap()[0];

        //Write uniform buffer to descriptor set
        let descriptor_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer.buffer)
            .range(uniform_buffer.size);

        let write_descriptor_set = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(slice::from_ref(&descriptor_buffer_info));

        unsafe { device.update_descriptor_sets(slice::from_ref(&write_descriptor_set), &[]) };

        Self {
            uniform_buffer,
            descriptor_set_layout,
            descriptor_set,
            device: device.clone(),
        }
    }

    pub fn update(&self, globals: &Globals) {
        unsafe {
            libc::memcpy(
                self.uniform_buffer.allocation_info.mapped_data,
                globals as *const _ as *const _,
                mem::size_of::<Globals>(),
            );
        }
    }
}
