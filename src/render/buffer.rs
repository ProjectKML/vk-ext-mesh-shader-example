use std::{mem, slice, sync::Arc};

use anyhow::Result;
use ash::{vk, Device};
use bytemuck::Pod;
use vk_mem_alloc::{Allocation, AllocationCreateFlags, AllocationCreateInfo, AllocationInfo, Allocator, MemoryUsage};

#[derive(Clone)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub allocation_info: AllocationInfo,
    pub device_address: vk::DeviceAddress,
    pub size: vk::DeviceSize,
    _device: Arc<Device>,
    allocator: Allocator
}

impl Buffer {
    pub unsafe fn new_uniform(device: Arc<Device>, allocator: Allocator, size: usize) -> Result<Self> {
        let (buffer, allocation, allocation_info) = vk_mem_alloc::create_buffer(
            allocator,
            &vk::BufferCreateInfo::default().size(size as _).usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
            &AllocationCreateInfo {
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
                usage: MemoryUsage::AUTO_PREFER_HOST,
                ..Default::default()
            }
        )?;

        Ok(Buffer {
            buffer,
            allocation,
            allocation_info,
            device_address: 0,
            size: size as _,
            _device: device,
            allocator
        })
    }

    pub unsafe fn new_device_local<T: Pod>(device: Arc<Device>, queue: vk::Queue, allocator: Allocator, data: &[T]) -> Result<Self> {
        let size = data.len() * mem::size_of::<T>();

        let (staging_buffer, staging_buffer_allocation, staging_buffer_allocation_info) = vk_mem_alloc::create_buffer(
            allocator,
            &vk::BufferCreateInfo::default().size(size as _).usage(vk::BufferUsageFlags::TRANSFER_SRC),
            &AllocationCreateInfo {
                flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | AllocationCreateFlags::MAPPED,
                usage: MemoryUsage::AUTO_PREFER_HOST,
                ..Default::default()
            }
        )?;

        libc::memcpy(staging_buffer_allocation_info.mapped_data.cast(), data.as_ptr().cast(), size);

        let (buffer, allocation, allocation_info) = vk_mem_alloc::create_buffer(
            allocator,
            &vk::BufferCreateInfo::default()
                .size(size as _)
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
            &AllocationCreateInfo {
                usage: MemoryUsage::AUTO_PREFER_DEVICE,
                ..Default::default()
            }
        )?;

        let device_address = device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

        let command_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default(), None)?;
        let command_buffer = device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(command_pool).command_buffer_count(1))?[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

        device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
        device.cmd_copy_buffer(command_buffer, staging_buffer, buffer, slice::from_ref(&vk::BufferCopy::default().size(size as _)));
        device.end_command_buffer(command_buffer).unwrap();

        device
            .queue_submit(queue, slice::from_ref(&vk::SubmitInfo::default().command_buffers(slice::from_ref(&command_buffer))), fence)
            .unwrap();

        device.wait_for_fences(slice::from_ref(&fence), true, u64::MAX).unwrap();

        device.destroy_fence(fence, None);
        device.free_command_buffers(command_pool, slice::from_ref(&command_buffer));
        device.destroy_command_pool(command_pool, None);

        vk_mem_alloc::destroy_buffer(allocator, staging_buffer, staging_buffer_allocation);

        Ok(Buffer {
            buffer,
            allocation,
            allocation_info,
            device_address,
            size: size as _,
            _device: device,
            allocator
        })
    }
}

impl Drop for Buffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            vk_mem_alloc::destroy_buffer(self.allocator, self.buffer, self.allocation);
        }
    }
}
