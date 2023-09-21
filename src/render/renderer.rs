use std::slice;

use ash::vk;
use glam::{Mat4, Quat, Vec3};
use winit::window::Window;

use crate::render::{
    render_ctx::{RenderCtx, FIELD_OF_VIEW},
    utils::globals::Globals,
};

unsafe fn update_globals(ctx: &RenderCtx, window: &Window) {
    //Compute view projection matrix
    let final_transform = &ctx.camera_rig.final_transform;

    let mut projection_matrix = Mat4::perspective_lh(
        FIELD_OF_VIEW.to_radians(),
        window.inner_size().width as f32 / window.inner_size().height as f32,
        0.1,
        1000.0,
    );
    projection_matrix.y_axis.y *= -1.0;

    let view_projection_matrix = projection_matrix
        * Mat4::look_at_lh(
            final_transform.position,
            final_transform.position + final_transform.forward(),
            final_transform.up(),
        )
        * Mat4::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0));

    ctx.globals_buffers.update(&Globals {
        view_projection_matrix,
        frustum_planes: Default::default(), //TODO:
        camera_pos: final_transform.position,
        time: 0.0, //TODO:
    })
}

pub fn render_frame(ctx: &RenderCtx, window: &Window, frame_index: &mut usize) {
    unsafe {
        //Begin frame
        let device_loader = &ctx.device_loader;
        let swapchain_loader = &ctx.swapchain_loader;

        let direct_queue = ctx.direct_queue;
        let swapchain = ctx.swapchain;

        let current_frame = &ctx.frames[*frame_index];

        let present_semaphore = current_frame.present_semaphore;
        let render_semaphore = current_frame.render_semaphore;

        let fence = current_frame.fence;
        device_loader
            .wait_for_fences(slice::from_ref(&fence), true, u64::MAX)
            .unwrap();
        device_loader.reset_fences(slice::from_ref(&fence)).unwrap();

        let command_pool = current_frame.command_pool;
        let command_buffer = current_frame.command_buffer;

        device_loader
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::RELEASE_RESOURCES)
            .unwrap();

        let image_index = swapchain_loader
            .acquire_next_image(swapchain, u64::MAX, present_semaphore, vk::Fence::null())
            .unwrap()
            .0;

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device_loader
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .unwrap();

        //Render frame
        update_globals(ctx, window);

        ctx.instance_cull_pass.execute(ctx, command_buffer);
        ctx.geometry_pass
            .execute(ctx, command_buffer, image_index as usize, window);

        //End frame
        device_loader.end_command_buffer(command_buffer).unwrap();

        let wait_semaphores = [present_semaphore];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_mask)
            .command_buffers(slice::from_ref(&command_buffer))
            .signal_semaphores(slice::from_ref(&render_semaphore));

        device_loader
            .queue_submit(direct_queue, slice::from_ref(&submit_info), fence)
            .unwrap();

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&render_semaphore))
            .swapchains(slice::from_ref(&swapchain))
            .image_indices(slice::from_ref(&image_index));

        swapchain_loader
            .queue_present(direct_queue, &present_info)
            .unwrap();
    }
}
