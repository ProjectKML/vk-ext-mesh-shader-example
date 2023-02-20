pub mod globals;
pub mod pipelines;

use std::slice;

use ash::{prelude::VkResult, vk, Device};
use vk_mem_alloc::{Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

#[inline]
pub unsafe fn create_descriptor_pool(
    device: &Device,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> VkResult<vk::DescriptorPool> {
    device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
            .max_sets(
                pool_sizes
                    .iter()
                    .map(|pool_size| pool_size.descriptor_count)
                    .sum(),
            )
            .pool_sizes(pool_sizes),
        None,
    )
}

unsafe fn change_image_layout(
    device: &Device,
    queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect_mask: vk::ImageAspectFlags,
) -> VkResult<()> {
    //Make image layout transition, we create and destroy command pool/buffer here to keep it simple
    let command_pool = device.create_command_pool(&vk::CommandPoolCreateInfo::default(), None)?;
    let command_buffer = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .command_buffer_count(1),
    )?[0];
    let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

    device.begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .level_count(1)
                .layer_count(1),
        );

    device.cmd_pipeline_barrier2(
        command_buffer,
        &vk::DependencyInfo::default()
            .image_memory_barriers(slice::from_ref(&image_memory_barrier)),
    );

    device.end_command_buffer(command_buffer)?;

    device.queue_submit(
        queue,
        slice::from_ref(
            &vk::SubmitInfo::default().command_buffers(slice::from_ref(&command_buffer)),
        ),
        fence,
    )?;

    //We wait for the fence and destroy all objects
    device.wait_for_fences(slice::from_ref(&fence), true, u64::MAX)?;
    device.destroy_fence(fence, None);
    device.destroy_command_pool(command_pool, None);

    Ok(())
}

pub unsafe fn create_depth_stencil_image(
    device: &Device,
    queue: vk::Queue,
    allocator: Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
) -> VkResult<(vk::Image, Allocation, vk::ImageView)> {
    let (image, allocation, _) = vk_mem_alloc::create_image(
        allocator,
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        &AllocationCreateInfo {
            usage: MemoryUsage::AUTO_PREFER_DEVICE,
            ..Default::default()
        },
    )?;

    let mut aspect_mask = vk::ImageAspectFlags::DEPTH;
    if format == vk::Format::D16_UNORM_S8_UINT
        || format == vk::Format::D24_UNORM_S8_UINT
        || format == vk::Format::D32_SFLOAT_S8_UINT
    {
        aspect_mask |= vk::ImageAspectFlags::STENCIL;
    }

    let image_view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(Default::default())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect_mask)
                    .level_count(1)
                    .layer_count(1),
            ),
        None,
    )?;

    change_image_layout(
        device,
        queue,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        aspect_mask,
    )?;

    Ok((image, allocation, image_view))
}

#[inline]
pub unsafe fn destroy_depth_stencil_image(
    device: &Device,
    allocator: Allocator,
    image: vk::Image,
    allocation: Allocation,
    image_view: vk::ImageView,
) {
    vk_mem_alloc::destroy_image(allocator, image, allocation);
    device.destroy_image_view(image_view, None);
}
