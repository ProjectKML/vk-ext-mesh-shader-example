use std::{ffi::CString, fs, fs::File, io::Read, mem, path::Path, slice};

use anyhow::{anyhow, Result};
use ash::{vk, Device};
use shaderc::{CompileOptions, Compiler, ResolvedInclude, ShaderKind, SpirvVersion};

fn create_shader_module(
    device: &Device,
    kind: ShaderKind,
    entry_point_name: &str,
    path: impl AsRef<Path>,
    defines: &[(&str, Option<&str>)],
) -> Result<vk::ShaderModule> {
    let path = path.as_ref();

    let mut file = File::open(path)?;

    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;

    let compiler = Compiler::new().ok_or_else(|| anyhow!("Failed to create compiler"))?;
    let mut compile_options =
        CompileOptions::new().ok_or_else(|| anyhow!("Failed to create compile options"))?;
    compile_options.set_include_callback(|requested_source, _, _, _| {
        let path = Path::new("shaders").join(requested_source); //TODO: Only working with simple paths, but it is ok for now

        Ok(ResolvedInclude {
            resolved_name: path.to_str().unwrap().to_owned(),
            content: fs::read_to_string(path).unwrap(),
        })
    });
    compile_options.set_target_spirv(SpirvVersion::V1_6);

    for (name, value) in defines {
        compile_options.add_macro_definition(name, *value);
    }

    let artifact = compiler.compile_into_spirv(
        &buffer,
        kind,
        path.to_str().unwrap(),
        entry_point_name,
        Some(&compile_options),
    )?;

    unsafe {
        let shader_module_create_info =
            vk::ShaderModuleCreateInfo::default().code(artifact.as_binary());

        Ok(device.create_shader_module(&shader_module_create_info, None)?)
    }
}

pub unsafe fn create_compute(
    device: &Device,
    path: impl AsRef<Path>,
    entry_point: &str,
    defines: &[(&str, Option<&str>)],
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let compute_shader =
        create_shader_module(device, ShaderKind::Compute, entry_point, path, defines)?;
    let entry_point = CString::new(entry_point)?;

    let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(compute_shader)
                .name(&entry_point),
        )
        .layout(layout);

    let pipeline = device
        .create_compute_pipelines(
            vk::PipelineCache::null(),
            slice::from_ref(&compute_pipeline_create_info),
            None,
        )
        .unwrap()[0];

    device.destroy_shader_module(compute_shader, None);

    Ok(pipeline)
}

pub unsafe fn create_mesh(
    device: &Device,
    mesh_path: impl AsRef<Path>,
    mesh_entry_point: &str,
    mesh_defines: &[(&str, Option<&str>)],
    fragment_path: impl AsRef<Path>,
    fragment_entry_point: &str,
    fragment_defines: &[(&str, Option<&str>)],
    swapchain_format: vk::Format,
    depth_format: vk::Format,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let mesh_shader = create_shader_module(
        device,
        ShaderKind::Mesh,
        mesh_entry_point,
        mesh_path,
        mesh_defines,
    )?;
    let fragment_shader = create_shader_module(
        device,
        ShaderKind::Fragment,
        fragment_entry_point,
        fragment_path,
        fragment_defines,
    )?;

    let mesh_entry_point = CString::new(mesh_entry_point)?;
    let fragment_entry_point = CString::new(fragment_entry_point)?;

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

    let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = vk::Viewport::default()
        .width(1.0)
        .height(1.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::default().extent(vk::Extent2D::default().width(1).height(1));

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(slice::from_ref(&viewport))
        .scissors(slice::from_ref(&scissor));

    let rasterization_state_create_info =
        vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);

    let depth_stencil_state_create_info = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let blend_attachment_state = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(slice::from_ref(&blend_attachment_state));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state_create_info =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

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

    let pipeline = device
        .create_graphics_pipelines(
            vk::PipelineCache::null(),
            slice::from_ref(&graphics_pipeline_create_info),
            None,
        )
        .unwrap()[0];

    device.destroy_shader_module(fragment_shader, None);
    device.destroy_shader_module(mesh_shader, None);

    Ok(pipeline)
}
