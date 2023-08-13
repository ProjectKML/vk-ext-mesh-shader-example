use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    mem, slice,
    sync::Arc,
};

use ash::{vk, Device};
use glam::{Quat, Vec3, Vec4};

use crate::render::{
    render_ctx::{RenderCtx, DEPTH_FORMAT, HEIGHT, SWAPCHAIN_FORMAT, WIDTH},
    utils,
    utils::globals::GlobalsBuffers,
};

pub struct GeometryPass {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub pipeline_tri: vk::Pipeline,
    pub triangle_view: bool,
    device: Arc<Device>,
}

impl Drop for GeometryPass {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline_tri, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl GeometryPass {
    pub fn new(
        device: &Arc<Device>,
        globals_buffers: &GlobalsBuffers,
        physical_device_mesh_shader_properties: &vk::PhysicalDeviceMeshShaderPropertiesEXT,
    ) -> Self {
        //Create descriptor set layout
        let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT);

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(slice::from_ref(&descriptor_set_layout_binding));

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }
        .unwrap();

        //Create pipeline layout
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::MESH_EXT)
            .size((mem::size_of::<Vec4>() * 2 + mem::size_of::<u32>() * 2) as _);

        let descriptor_set_layouts = [globals_buffers.descriptor_set_layout, descriptor_set_layout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(slice::from_ref(&push_constant_range));
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap();

        //Create pipeline
        let (pipeline, pipeline_tri) = unsafe {
            let local_size_x = physical_device_mesh_shader_properties
                .max_preferred_mesh_work_group_invocations
                .to_string();

            (
                utils::pipelines::create_mesh(
                    device,
                    "shaders/geometry.mesh.glsl",
                    "main",
                    &[("LOCAL_SIZE_X", Some(&local_size_x))],
                    "shaders/geometry.frag.glsl",
                    "main",
                    &[],
                    SWAPCHAIN_FORMAT,
                    DEPTH_FORMAT,
                    pipeline_layout,
                )
                .unwrap(),
                utils::pipelines::create_mesh(
                    device,
                    "shaders/geometry_tri.mesh.glsl",
                    "main",
                    &[("LOCAL_SIZE_X", Some(&local_size_x))],
                    "shaders/geometry_tri.frag.glsl",
                    "main",
                    &[],
                    SWAPCHAIN_FORMAT,
                    DEPTH_FORMAT,
                    pipeline_layout,
                )
                .unwrap(),
            )
        };

        Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            pipeline_tri,
            triangle_view: false,
            device: device.clone(),
        }
    }

    pub unsafe fn execute(
        &self,
        ctx: &RenderCtx,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        let device_loader = &ctx.device_loader;

        let image = ctx.swapchain_images[image_index];

        //Transition image to COLOR_ATTACHMENT_OPTIMAL
        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        device_loader.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(slice::from_ref(&image_memory_barrier)),
        );

        //Begin rendering
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(ctx.swapchain_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [100.0 / 255.0, 149.0 / 255.0, 237.0 / 255.0, 1.0],
                },
            });

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(ctx.depth_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let rendering_info = vk::RenderingInfo::default()
            .render_area(
                vk::Rect2D::default().extent(vk::Extent2D::default().width(WIDTH).height(HEIGHT)),
            )
            .layer_count(1)
            .color_attachments(slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        device_loader.cmd_begin_rendering(command_buffer, &rendering_info);

        //Bind pipeline, set viewport and bind descriptor set
        ctx.device_loader.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            if self.triangle_view {
                self.pipeline_tri
            } else {
                self.pipeline
            },
        );

        let viewport = vk::Viewport::default()
            .width(WIDTH as _)
            .height(HEIGHT as _)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        });

        ctx.device_loader
            .cmd_set_viewport(command_buffer, 0, slice::from_ref(&viewport));

        ctx.device_loader
            .cmd_set_scissor(command_buffer, 0, slice::from_ref(&scissor));

        ctx.device_loader.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            ctx.geometry_pass.pipeline_layout,
            0,
            &[
                ctx.globals_buffers.descriptor_set,
                ctx.mesh_collection.descriptor_set,
            ],
            &[],
        );

        //Execute draw
        render_meshes(ctx, command_buffer);

        //End rendering
        device_loader.cmd_end_rendering(command_buffer);

        //Transition image to PRESENT_SRC_KHR
        let image_memory_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );

        device_loader.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .image_memory_barriers(slice::from_ref(&image_memory_barrier)),
        );
    }
}

unsafe fn render_meshes(ctx: &RenderCtx, command_buffer: vk::CommandBuffer) {
    ctx.mesh_collection.draw_mesh(
        ctx,
        command_buffer,
        &Vec3::new(-120.43, -2.325, -160.1),
        280.20,
        &Quat::IDENTITY,
        0,
        0,
    );

    for i in 0..25 {
        for j in 0..25 {
            let angle = {
                let mut hasher = DefaultHasher::new();
                (i * 1128889).hash(&mut hasher);
                (j * 1254739).hash(&mut hasher);
                (i + j).hash(&mut hasher);

                let hash_code = hasher.finish();
                (hash_code & 255) as f32 / 255.0 * std::f32::consts::PI
            };

            let mesh_idx = ((i + j) % 4) + 1;
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

            let final_transform = &ctx.camera_rig.final_transform;

            let level_idx = (((final_transform.position.distance(rotation * translation)) * 0.08)
                as u32)
                .min(max_level_idx as _);
            unsafe {
                ctx.mesh_collection.draw_mesh(
                    ctx,
                    command_buffer,
                    &translation,
                    scale as _,
                    &rotation,
                    mesh_idx as _,
                    level_idx,
                )
            };
        }
    }
}
