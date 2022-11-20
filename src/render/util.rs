use std::{ffi::CString, fs::File, io::Read, mem, path::Path, slice};

use anyhow::{anyhow, Result};
use ash::{prelude::VkResult, vk, Device};
use shaderc::{CompileOptions, Compiler, ShaderKind, SpirvVersion};
use vk_mem_alloc::{Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

#[inline]
pub unsafe fn create_descriptor_pool(device: &Device, pool_sizes: &[vk::DescriptorPoolSize]) -> VkResult<vk::DescriptorPool> {
    device.create_descriptor_pool(
        &vk::DescriptorPoolCreateInfo::default()
            .max_sets(pool_sizes.iter().map(|pool_size| pool_size.descriptor_count).sum())
            .pool_sizes(pool_sizes),
        None
    )
}

#[inline]
pub fn create_shader_module(device: &Device, kind: ShaderKind, entry_point_name: &str, path: impl AsRef<Path>) -> Result<vk::ShaderModule> {
    let mut file = File::open(path)?;

    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;

    let compiler = Compiler::new().ok_or_else(|| anyhow!("Failed to create compiler"))?;
    let mut compile_options = CompileOptions::new().ok_or_else(|| anyhow!("Failed to create compile options"))?;
    compile_options.set_target_spirv(SpirvVersion::V1_6);

    let artifact = compiler.compile_into_spirv(&buffer, kind, "", entry_point_name, Some(&compile_options))?;

    unsafe {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(artifact.as_binary());

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
    depth_format: vk::Format,
    layout: vk::PipelineLayout,
    mesh_shader_properties: &vk::PhysicalDeviceMeshShaderPropertiesEXT
) -> Result<vk::Pipeline> {
    let mesh_entry_point = CString::new(mesh_entry_point).unwrap();
    let fragment_entry_point = CString::new(fragment_entry_point).unwrap();

    let specialization_map_entry = vk::SpecializationMapEntry::default().size(mem::size_of::<u32>());

    let values = [mesh_shader_properties.max_preferred_mesh_work_group_invocations];

    let specialization_info = vk::SpecializationInfo::default()
        .map_entries(slice::from_ref(&specialization_map_entry))
        .data(bytemuck::cast_slice(&values));

    let shader_stage_create_infos = vec![
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::MESH_EXT)
            .module(mesh_shader)
            .name(&mesh_entry_point)
            .specialization_info(&specialization_info),
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

    let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::default();

    let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_attachment_state = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::default().attachments(slice::from_ref(&blend_attachment_state));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(slice::from_ref(&swapchain_format))
        .depth_attachment_format(depth_format);

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

pub unsafe fn create_depth_stencil_image(device: &Device, allocator: Allocator, width: u32, height: u32, format: vk::Format) -> VkResult<(vk::Image, Allocation, vk::ImageView)> {
    let (image, allocation, _) = vk_mem_alloc::create_image(
        allocator,
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        &AllocationCreateInfo {
            usage: MemoryUsage::AUTO_PREFER_DEVICE,
            ..Default::default()
        }
    )?;

    let mut aspect_mask = vk::ImageAspectFlags::DEPTH;
    if format == vk::Format::D16_UNORM_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT || format == vk::Format::D32_SFLOAT_S8_UINT {
        aspect_mask |= vk::ImageAspectFlags::STENCIL;
    }

    let image_view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(Default::default())
            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(aspect_mask).level_count(1).layer_count(1)),
        None
    )?;

    Ok((image, allocation, image_view))
}

#[inline]
pub unsafe fn destroy_depth_stencil_image(device: &Device, allocator: Allocator, image: vk::Image, allocation: Allocation, image_view: vk::ImageView) {
    vk_mem_alloc::destroy_image(allocator, image, allocation);
    device.destroy_image_view(image_view, None);
}
