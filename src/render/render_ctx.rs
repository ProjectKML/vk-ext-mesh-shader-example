use std::{mem, mem::ManuallyDrop, slice, sync::Arc};

use ash::{
    extensions::{
        ext::MeshShader,
        khr::{Surface, Swapchain},
    },
    vk, Device, Entry, Instance,
};
use dolly::{
    drivers::Position,
    prelude::{CameraRig, Smooth, YawPitch},
};
use glam::{Vec2, Vec3, Vec4};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use shaderc::ShaderKind;
use vk_mem_alloc::{Allocation, AllocatorCreateFlags, AllocatorCreateInfo};
use winit::window::Window;

use crate::render::{
    frame,
    frame::Frame,
    mesh::{MeshCollection, MeshSource, Vertex},
    query_pool::QueryPool,
    utils,
};

pub const WIDTH: u32 = 1600;
pub const HEIGHT: u32 = 900;
pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const FIELD_OF_VIEW: f32 = 90.0;

pub struct RenderCtx {
    pub entry_loader: Entry,

    pub instance_loader: Instance,
    pub surface_loader: Surface,

    pub surface: vk::SurfaceKHR,

    pub device_loader: Arc<Device>,
    pub swapchain_loader: Swapchain,
    pub mesh_shader_loader: MeshShader,

    pub allocator: vk_mem_alloc::Allocator,

    pub direct_queue: vk::Queue,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    pub depth_image_allocation: Allocation,

    pub descriptor_pool: vk::DescriptorPool,

    pub geometry_descriptor_set_layout: vk::DescriptorSetLayout,
    pub geometry_pipeline_layout: vk::PipelineLayout,
    pub geometry_pipeline: vk::Pipeline,

    pub frames: Vec<ManuallyDrop<Frame>>,
    pub camera_rig: CameraRig,
    pub mesh_collection: ManuallyDrop<MeshCollection>,

    pub query_pool_timestamp: ManuallyDrop<QueryPool>,
    pub query_pool_pipeline_statistics: ManuallyDrop<QueryPool>,

    pub workgroup_size: u32,
}

fn create_geometry_pipeline(
    device_loader: &Device,
    physical_device_mesh_shader_properties: &vk::PhysicalDeviceMeshShaderPropertiesEXT,
) -> (vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline) {
    //Create descriptor set layout
    let descriptor_set_layout_bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::MESH_EXT),
    ];

    let descriptor_set_layout = unsafe {
        device_loader.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings),
            None,
        )
    }
    .unwrap();

    //Create pipeline layout
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::MESH_EXT)
        .size((mem::size_of::<Vec4>() * 2 + mem::size_of::<u32>() * 2) as _);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(slice::from_ref(&push_constant_range));
    let pipeline_layout =
        unsafe { device_loader.create_pipeline_layout(&pipeline_layout_create_info, None) }
            .unwrap();

    //Create pipeline
    let pipeline = unsafe {
        let local_size_x = physical_device_mesh_shader_properties
            .max_preferred_mesh_work_group_invocations
            .to_string();

        utils::pipelines::create_mesh(
            &device_loader,
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
    }
    .unwrap();

    (descriptor_set_layout, pipeline_layout, pipeline)
}

unsafe fn destroy_geometry_pipeline(
    device_loader: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
) {
    device_loader.destroy_pipeline(pipeline, None);
    device_loader.destroy_pipeline_layout(pipeline_layout, None);

    device_loader.destroy_descriptor_set_layout(descriptor_set_layout, None);
}

fn create_instance_pipeline() -> (vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline) {
    todo!()
}

unsafe fn destroy_instance_cull_pipeline(
    device_loader: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
) {
    todo!()
}

impl RenderCtx {
    pub fn new(window: &Window) -> Self {
        let entry_loader = unsafe { Entry::load() }.unwrap();

        let application_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let instance_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr().cast()];

        let mut instance_extensions = vec![];
        ash_window::enumerate_required_extensions(window.raw_display_handle())
            .unwrap()
            .iter()
            .for_each(|e| instance_extensions.push(*e));

        let instance_create_info = vk::InstanceCreateInfo::default()
            .enabled_layer_names(&instance_layers)
            .enabled_extension_names(&instance_extensions)
            .application_info(&application_info);

        let instance_loader =
            unsafe { entry_loader.create_instance(&instance_create_info, None) }.unwrap();
        let surface_loader = Surface::new(&entry_loader, &instance_loader);

        let surface = unsafe {
            ash_window::create_surface(
                &entry_loader,
                &instance_loader,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }
        .unwrap();

        let physical_devices = unsafe { instance_loader.enumerate_physical_devices() }.unwrap();
        let physical_device = physical_devices[0];

        let mut physical_device_vulkan_12_properties =
            vk::PhysicalDeviceVulkan12Properties::default();
        let mut physical_device_vulkan_13_properties =
            vk::PhysicalDeviceVulkan13Properties::default();
        let mut physical_device_mesh_shader_properties =
            vk::PhysicalDeviceMeshShaderPropertiesEXT::default();

        let mut physical_device_properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut physical_device_vulkan_12_properties)
            .push_next(&mut physical_device_vulkan_13_properties)
            .push_next(&mut physical_device_mesh_shader_properties);

        unsafe {
            instance_loader
                .get_physical_device_properties2(physical_device, &mut physical_device_properties)
        };

        dbg!(&physical_device_mesh_shader_properties);

        let queue_priority = 1.0;
        let device_queue_create_info =
            vk::DeviceQueueCreateInfo::default().queue_priorities(slice::from_ref(&queue_priority));

        let device_extensions = [Swapchain::name().as_ptr(), MeshShader::name().as_ptr()];

        let physical_device_features =
            vk::PhysicalDeviceFeatures::default().pipeline_statistics_query(true);

        let mut physical_device_vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
        let mut physical_device_vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let mut physical_device_mesh_shader_features =
            vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);

        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .features(physical_device_features)
            .push_next(&mut physical_device_vulkan_12_features)
            .push_next(&mut physical_device_vulkan_13_features)
            .push_next(&mut physical_device_mesh_shader_features);

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut physical_device_features)
            .queue_create_infos(slice::from_ref(&device_queue_create_info))
            .enabled_extension_names(&device_extensions);
        let device_loader = Arc::new(
            unsafe { instance_loader.create_device(physical_device, &device_create_info, None) }
                .unwrap(),
        );
        let swapchain_loader = Swapchain::new(&instance_loader, &device_loader);
        let mesh_shader_loader = MeshShader::new(&instance_loader, &device_loader);

        let allocator = unsafe {
            vk_mem_alloc::create_allocator(
                &instance_loader,
                physical_device,
                &device_loader,
                Some(&AllocatorCreateInfo {
                    flags: AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS,
                    ..Default::default()
                }),
            )
        }
        .unwrap();

        let direct_queue = unsafe { device_loader.get_device_queue(0, 0) };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(SWAPCHAIN_FORMAT)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D {
                width: WIDTH,
                height: HEIGHT,
            })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain =
            unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None) }.unwrap();
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(SWAPCHAIN_FORMAT)
                    .components(Default::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1),
                    );

                unsafe { device_loader.create_image_view(&image_view_create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let (depth_image, depth_image_allocation, depth_image_view) = unsafe {
            utils::create_depth_stencil_image(
                &device_loader,
                direct_queue,
                allocator,
                WIDTH,
                HEIGHT,
                DEPTH_FORMAT,
            )
        }
        .unwrap();

        let descriptor_pool = unsafe {
            utils::create_descriptor_pool(
                &device_loader,
                &[vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(3)],
            )
        }
        .unwrap();

        let (geometry_descriptor_set_layout, geometry_pipeline_layout, geometry_pipeline) =
            create_geometry_pipeline(&device_loader, &physical_device_mesh_shader_properties);

        let frames: Vec<_> = (0..frame::NUM_FRAMES)
            .into_iter()
            .map(|_| ManuallyDrop::new(Frame::new(device_loader.clone())))
            .collect();

        let camera_rig = CameraRig::builder()
            .with(Position::new(Vec3::Y))
            .with(YawPitch::new())
            .with(Smooth::new_position_rotation(1.0, 1.0))
            .build();

        let mesh_collection = ManuallyDrop::new(
            unsafe {
                MeshCollection::new(
                    &device_loader,
                    direct_queue,
                    allocator,
                    descriptor_pool,
                    geometry_descriptor_set_layout,
                    [
                        MeshSource::Builtin(
                            vec![
                                Vertex::new(
                                    Vec3::new(0.0, 0.0, 0.0),
                                    Vec2::new(0.0, 0.0),
                                    Vec3::new(0.0, 1.0, 0.0),
                                ),
                                Vertex::new(
                                    Vec3::new(1.0, 0.0, 0.0),
                                    Vec2::new(1.0, 0.0),
                                    Vec3::new(0.0, 1.0, 0.0),
                                ),
                                Vertex::new(
                                    Vec3::new(1.0, 0.0, 1.0),
                                    Vec2::new(1.0, 1.0),
                                    Vec3::new(0.0, 1.0, 0.0),
                                ),
                                Vertex::new(
                                    Vec3::new(0.0, 0.0, 1.0),
                                    Vec2::new(0.0, 1.0),
                                    Vec3::new(0.0, 1.0, 0.0),
                                ),
                            ],
                            vec![0, 1, 3, 3, 1, 2],
                        ),
                        MeshSource::Path("dragon.obj"),
                        MeshSource::Path("armadillo.obj"),
                        MeshSource::Path("bunny.obj"),
                    ],
                )
            }
            .unwrap(),
        );

        let query_pool_timestamp = ManuallyDrop::new(
            unsafe { QueryPool::new(&device_loader, 8, vk::QueryType::TIMESTAMP) }.unwrap(),
        );
        let query_pool_pipeline_statistics = ManuallyDrop::new(
            unsafe { QueryPool::new(&device_loader, 8, vk::QueryType::PIPELINE_STATISTICS) }
                .unwrap(),
        );

        Self {
            entry_loader,

            instance_loader,
            surface_loader,

            surface,

            device_loader,
            swapchain_loader,
            mesh_shader_loader,

            allocator,

            direct_queue,

            swapchain,
            swapchain_images,
            swapchain_image_views,
            depth_image,
            depth_image_view,
            depth_image_allocation,

            descriptor_pool,

            geometry_descriptor_set_layout,
            geometry_pipeline_layout,
            geometry_pipeline,

            frames,
            camera_rig,
            mesh_collection,

            query_pool_timestamp,
            query_pool_pipeline_statistics,

            workgroup_size: physical_device_mesh_shader_properties
                .max_preferred_mesh_work_group_invocations,
        }
    }
}

impl Drop for RenderCtx {
    fn drop(&mut self) {
        unsafe {
            self.device_loader.device_wait_idle().unwrap();

            ManuallyDrop::drop(&mut self.query_pool_pipeline_statistics);
            ManuallyDrop::drop(&mut self.query_pool_timestamp);

            ManuallyDrop::drop(&mut self.mesh_collection);
            self.frames
                .iter_mut()
                .for_each(|frame| ManuallyDrop::drop(frame));

            destroy_geometry_pipeline(
                &self.device_loader,
                self.geometry_descriptor_set_layout,
                self.geometry_pipeline_layout,
                self.geometry_pipeline,
            );

            self.device_loader
                .destroy_descriptor_pool(self.descriptor_pool, None);

            utils::destroy_depth_stencil_image(
                &self.device_loader,
                self.allocator,
                self.depth_image,
                self.depth_image_allocation,
                self.depth_image_view,
            );
            self.swapchain_image_views
                .iter()
                .for_each(|image_view| self.device_loader.destroy_image_view(*image_view, None));
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            vk_mem_alloc::destroy_allocator(self.allocator);

            self.device_loader.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance_loader.destroy_instance(None);
        }
    }
}
