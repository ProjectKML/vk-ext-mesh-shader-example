use std::{mem::ManuallyDrop, slice, sync::Arc};

use ash::{
    extensions::{
        ext::MeshShader,
        khr::{DynamicRendering, Surface, Swapchain}
    },
    vk, Device, Entry, Instance
};
use winit::window::Window;

use crate::render::{frame, frame::Frame, util};

pub const WIDTH: u32 = 1600;
pub const HEIGHT: u32 = 900;
pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

pub struct RenderCtx {
    pub entry_loader: Entry,

    pub instance_loader: Instance,
    pub surface_loader: Surface,

    pub surface: vk::SurfaceKHR,

    pub device_loader: Arc<Device>,
    pub swapchain_loader: Swapchain,
    pub dynamic_rendering_loader: DynamicRendering,
    pub mesh_shader_loader: MeshShader,

    pub direct_queue: vk::Queue,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,

    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub frames: Vec<ManuallyDrop<Frame>>
}

impl RenderCtx {
    pub fn new(window: &Window) -> Self {
        let entry_loader = unsafe { Entry::load() }.unwrap();

        let application_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);

        let instance_layers = [b"VK_LAYER_KHRONOS_validation\0".as_ptr().cast()];

        let mut instance_extensions = vec![];
        ash_window::enumerate_required_extensions(&window).unwrap().iter().for_each(|e| instance_extensions.push(*e));

        let instance_create_info = vk::InstanceCreateInfo::default()
            .enabled_layer_names(&instance_layers)
            .enabled_extension_names(&instance_extensions)
            .application_info(&application_info);

        let instance_loader = unsafe { entry_loader.create_instance(&instance_create_info, None) }.unwrap();
        let surface_loader = Surface::new(&entry_loader, &instance_loader);

        let surface = unsafe { ash_window::create_surface(&entry_loader, &instance_loader, &window, None) }.unwrap();

        let physical_devices = unsafe { instance_loader.enumerate_physical_devices() }.unwrap();
        let physical_device = physical_devices[0];

        let queue_priority = 1.0;
        let device_queue_create_info = vk::DeviceQueueCreateInfo::default().queue_priorities(slice::from_ref(&queue_priority));

        let device_extensions = [Swapchain::name().as_ptr(), DynamicRendering::name().as_ptr(), MeshShader::name().as_ptr()];

        let physical_device_features = vk::PhysicalDeviceFeatures::default();

        let mut physical_device_dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut physical_device_mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);

        let mut physical_device_features = vk::PhysicalDeviceFeatures2::default()
            .features(physical_device_features)
            .push_next(&mut physical_device_dynamic_rendering_features)
            .push_next(&mut physical_device_mesh_shader_features);

        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut physical_device_features)
            .queue_create_infos(slice::from_ref(&device_queue_create_info))
            .enabled_extension_names(&device_extensions);
        let device_loader = Arc::new(unsafe { instance_loader.create_device(physical_device, &device_create_info, None) }.unwrap());
        let swapchain_loader = Swapchain::new(&instance_loader, &device_loader);
        let dynamic_rendering_loader = DynamicRendering::new(&instance_loader, &device_loader);
        let mesh_shader_loader = MeshShader::new(&instance_loader, &device_loader);

        let direct_queue = unsafe { device_loader.get_device_queue(0, 0) };

        let swapchian_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(2)
            .image_format(SWAPCHAIN_FORMAT)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D { width: WIDTH, height: HEIGHT })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchian_create_info, None) }.unwrap();
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| {
                let image_view_create_info = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(SWAPCHAIN_FORMAT)
                    .components(Default::default())
                    .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1).level_count(1));

                unsafe { device_loader.create_image_view(&image_view_create_info, None) }
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let pipeline_layout = unsafe { device_loader.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None) }.unwrap();

        let mesh_shader = util::create_shader_module(&device_loader, "example.mesh.spv").unwrap();
        let fragment_shader = util::create_shader_module(&device_loader, "example.frag.spv").unwrap();

        let pipeline = unsafe { util::create_mesh_pipeline(&device_loader, mesh_shader, "main", fragment_shader, "main", SWAPCHAIN_FORMAT, pipeline_layout) }.unwrap();

        unsafe { device_loader.destroy_shader_module(fragment_shader, None) };
        unsafe { device_loader.destroy_shader_module(mesh_shader, None) };

        let frames: Vec<_> = (0..frame::NUM_FRAMES).into_iter().map(|_| ManuallyDrop::new(Frame::new(device_loader.clone()))).collect();

        Self {
            entry_loader,

            instance_loader,
            surface_loader,

            surface,

            device_loader,
            swapchain_loader,
            dynamic_rendering_loader,
            mesh_shader_loader,

            direct_queue,

            swapchain,
            swapchain_images,
            swapchain_image_views,

            pipeline_layout,
            pipeline,

            frames
        }
    }
}

impl Drop for RenderCtx {
    fn drop(&mut self) {
        unsafe {
            self.device_loader.device_wait_idle().unwrap();

            self.frames.iter_mut().for_each(|frame| ManuallyDrop::drop(frame));

            self.device_loader.destroy_pipeline(self.pipeline, None);
            self.device_loader.destroy_pipeline_layout(self.pipeline_layout, None);

            self.swapchain_image_views.iter().for_each(|image_view| self.device_loader.destroy_image_view(*image_view, None));
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.device_loader.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance_loader.destroy_instance(None);
        }
    }
}
