use std::{ffi::CString, fs::File, io::Read, path::Path, slice};

use anyhow::Result;
use ash::{prelude::VkResult, vk, Device};

pub unsafe fn create_descriptor_pool(device: &Device, pool_sizes: &[vk::DescriptorPoolSize]) -> VkResult<vk::DescriptorPool> {
    device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
            .max_sets(pool_sizes.iter().map(|pool_size| pool_size.descriptor_count).sum())
            .pool_sizes(pool_sizes),
        None
    )
}

pub fn create_shader_module(device: &Device, path: impl AsRef<Path>) -> Result<vk::ShaderModule> {
    let mut file = File::open(path)?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(slice::from_raw_parts(buffer.as_ptr().cast(), buffer.len() >> 2));

        Ok(device.create_shader_module(&shader_module_create_info, None)?)
    }
}

pub unsafe fn create_mesh_pipeline(
    device: &Device,
    mesh_shader: vk::ShaderModule,
    mesh_entry_point: &str,
    fragment_shader: vk::ShaderModule,
    fragment_entry_point: &str,
    swapchain_format: vk::Format,
    layout: vk::PipelineLayout
) -> Result<vk::Pipeline> {
    let mesh_entry_point = CString::new(mesh_entry_point).unwrap();
    let fragment_entry_point = CString::new(fragment_entry_point).unwrap();

    let shader_stage_create_infos = vec![
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .module(mesh_shader)
            .name(&mesh_entry_point),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader)
            .name(&fragment_entry_point),
    ];

    let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default().width(1.0).height(1.0).max_depth(1.0);

    let scissor = vk::Rect2D::default().extent(vk::Extent2D { width: 1, height: 1 });

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(slice::from_ref(&viewport))
        .scissors(slice::from_ref(&scissor));

    let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);

    let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_attachment_state = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::default().attachments(slice::from_ref(&blend_attachment_state));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(slice::from_ref(&swapchain_format));

    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stage_create_infos)
        .input_assembly_state(&input_assembly_state_create_info)
        .viewport_state(&viewport_state_create_info)
        .rasterization_state(&rasterization_state_create_info)
        .depth_stencil_state(&depth_stencil_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .color_blend_state(&color_blend_state_create_info)
        .dynamic_state(&dynamic_state_create_info)
        .layout(layout)
        .push_next(&mut pipeline_rendering_create_info);

    Ok(device
        .create_graphics_pipelines(vk::PipelineCache::null(), slice::from_ref(&graphics_pipeline_create_info), None)
        .unwrap()[0])
}
