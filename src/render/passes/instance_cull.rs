use std::sync::Arc;

use ash::{vk, Device};

use crate::render::{
    passes::geometry::GeometryPass, render_ctx::RenderCtx, utils, utils::globals::GlobalsBuffers,
};

pub struct InstanceCullPass {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl Drop for InstanceCullPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl InstanceCullPass {
    pub fn new(
        device: &Arc<Device>,
        globals_buffers: &GlobalsBuffers,
        geometry_pass: &GeometryPass,
    ) -> Self {
        //Create descriptor set layout
        let descriptor_set_layout_bindings = (0..4)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        }
        .unwrap();

        //Create pipeline layout
        let descriptor_set_layouts = [
            globals_buffers.descriptor_set_layout,
            geometry_pass.descriptor_set_layout,
            descriptor_set_layout,
        ];

        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }.unwrap();

        //Create pipeline
        let pipeline = unsafe {
            utils::pipelines::create_compute(
                device,
                "shaders/instance_cull.comp.glsl",
                "main",
                &[],
                pipeline_layout,
            )
        }
        .unwrap();

        Self {
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            device: device.clone(),
        }
    }

    pub fn execute(&self, _ctx: &RenderCtx, _command_buffer: vk::CommandBuffer) {
        //TODO:
    }
}
