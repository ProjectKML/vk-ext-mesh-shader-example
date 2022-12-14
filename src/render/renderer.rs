use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    slice
};

use ash::vk;
use glam::{Mat4, Quat, Vec3};

use crate::render::{
    frame::Frame,
    render_ctx::{RenderCtx, FIELD_OF_VIEW, HEIGHT, WIDTH}
};

pub fn render_frame(ctx: &RenderCtx, frame_index: &mut usize) {
    let device_loader = &ctx.device_loader;
    let swapchain_loader = &ctx.swapchain_loader;

    let direct_queue = ctx.direct_queue;
    let swapchain = ctx.swapchain;

    let current_frame = &ctx.frames[*frame_index];

    let present_semaphore = current_frame.present_semaphore;
    let render_semaphore = current_frame.render_semaphore;

    let fence = current_frame.fence;
    unsafe { device_loader.wait_for_fences(slice::from_ref(&fence), true, u64::MAX) }.unwrap();
    unsafe { device_loader.reset_fences(slice::from_ref(&fence)) }.unwrap();

    let command_pool = current_frame.command_pool;
    let command_buffer = current_frame.command_buffer;

    unsafe { device_loader.reset_command_pool(command_pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES) }.unwrap();

    let image_index = unsafe { swapchain_loader.acquire_next_image(swapchain, u64::MAX, present_semaphore, vk::Fence::null()) }
        .unwrap()
        .0;

    let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device_loader.begin_command_buffer(command_buffer, &command_buffer_begin_info) }.unwrap();

    let image = ctx.swapchain_images[image_index as usize];

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

    unsafe {
        device_loader.cmd_pipeline_barrier2(command_buffer, &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_memory_barrier)));
    }

    let color_attachment = vk::RenderingAttachmentInfo::default()
        .image_view(ctx.swapchain_image_views[image_index as usize])
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [100.0 / 255.0, 149.0 / 255.0, 237.0 / 255.0, 1.0]
            }
        });

    let depth_attachment = vk::RenderingAttachmentInfo::default()
        .image_view(ctx.depth_image_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }
        });

    let rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D::default().extent(vk::Extent2D::default().width(WIDTH).height(HEIGHT)))
        .layer_count(1)
        .color_attachments(slice::from_ref(&color_attachment))
        .depth_attachment(&depth_attachment);

    unsafe { device_loader.cmd_begin_rendering(command_buffer, &rendering_info) };

    render_frame_inner(ctx, current_frame);

    unsafe { device_loader.cmd_end_rendering(command_buffer) };

    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));

    unsafe {
        device_loader.cmd_pipeline_barrier2(command_buffer, &vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_memory_barrier)));
    };

    unsafe { device_loader.end_command_buffer(command_buffer) }.unwrap();

    let wait_semaphores = [present_semaphore];
    let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

    let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_dst_stage_mask)
        .command_buffers(slice::from_ref(&command_buffer))
        .signal_semaphores(slice::from_ref(&render_semaphore));

    unsafe { device_loader.queue_submit(direct_queue, slice::from_ref(&submit_info), fence) }.unwrap();

    let present_info = vk::PresentInfoKHR::default()
        .wait_semaphores(slice::from_ref(&render_semaphore))
        .swapchains(slice::from_ref(&swapchain))
        .image_indices(slice::from_ref(&image_index));

    unsafe { swapchain_loader.queue_present(direct_queue, &present_info) }.unwrap();
}

fn render_frame_inner(ctx: &RenderCtx, current_frame: &Frame) {
    let command_buffer = current_frame.command_buffer;

    unsafe { ctx.device_loader.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, ctx.pipeline) };

    let viewport = vk::Viewport::default().width(WIDTH as _).height(HEIGHT as _).max_depth(1.0);
    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width: WIDTH, height: HEIGHT });

    unsafe { ctx.device_loader.cmd_set_viewport(command_buffer, 0, slice::from_ref(&viewport)) };
    unsafe { ctx.device_loader.cmd_set_scissor(command_buffer, 0, slice::from_ref(&scissor)) };

    let final_transform = &ctx.camera_rig.final_transform;

    let mut projection_matrix = Mat4::perspective_lh(FIELD_OF_VIEW.to_radians(), WIDTH as f32 / HEIGHT as f32, 0.1, 1000.0);
    projection_matrix.y_axis.y *= -1.0;

    let view_projection_matrix = projection_matrix
        * Mat4::look_at_lh(final_transform.position, final_transform.position + final_transform.forward(), final_transform.up())
        * Mat4::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0));

    unsafe { ctx.mesh_collection.bind(ctx, command_buffer, &view_projection_matrix, &final_transform.position) };

    unsafe {
        ctx.mesh_collection
            .draw_mesh(ctx, command_buffer, &Vec3::new(-120.43, -2.325, -160.1), 280.20, &Quat::IDENTITY, 0, 0)
    };

    for i in 0..13 {
        for j in 0..25 {
            let angle = {
                let mut hasher = DefaultHasher::new();
                (i * 1128889).hash(&mut hasher);
                (j * 1254739).hash(&mut hasher);
                (i + j).hash(&mut hasher);

                let hash_code = hasher.finish();
                (hash_code & 255) as f32 / 255.0 * std::f32::consts::PI
            };

            let mesh_idx = ((i + j) % 3) + 1;
            let (scale, y_offset) = if mesh_idx == 1 {
                (1.0, -2.6)
            } else if mesh_idx == 2 {
                (0.1, 2.8)
            } else {
                (22.0, -3.25)
            };

            let translation = Vec3::new(i as f32 * 7.0, y_offset, j as f32 * 5.0);
            let rotation = Quat::from_rotation_y(angle);

            let max_level_idx = ctx.mesh_collection.mesh_buffers_at(mesh_idx).levels.len();

            let level_idx = (((final_transform.position.distance(rotation * translation)) * 0.08) as u32).min(max_level_idx as _);
            unsafe { ctx.mesh_collection.draw_mesh(ctx, command_buffer, &translation, scale as _, &rotation, mesh_idx as _, level_idx) };
        }
    }
}
