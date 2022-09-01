use std::{slice, sync::Arc};

use ash::{vk, Device};

pub const NUM_FRAMES: usize = 2;

pub struct Frame {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub present_semaphore: vk::Semaphore,
    pub render_semaphore: vk::Semaphore,

    pub fence: vk::Fence,

    device: Arc<Device>
}

impl Frame {
    pub fn new(device: Arc<Device>) -> Self {
        let command_pool = unsafe { device.create_command_pool(&vk::CommandPoolCreateInfo::default(), None) }.unwrap();
        let command_buffer = unsafe { device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(command_pool).command_buffer_count(1)) }.unwrap()[0];
        let present_semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();
        let render_semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.unwrap();
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None) }.unwrap();

        Self {
            command_pool,
            command_buffer,
            present_semaphore,
            render_semaphore,
            fence,
            device
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);

            self.device.destroy_semaphore(self.render_semaphore, None);
            self.device.destroy_semaphore(self.present_semaphore, None);

            self.device.free_command_buffers(self.command_pool, slice::from_ref(&self.command_buffer));
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
