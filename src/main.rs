use std::collections::HashSet;

use dolly::prelude::{Position, YawPitch};
use glam::Vec3;
use winit::{
    dpi::{PhysicalSize, Size},
    event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::WindowBuilder
};

use crate::render::{render_ctx::RenderCtx, renderer};

pub mod render;

fn main() {
    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("vk-ext-mesh-shader-example")
        .with_inner_size(Size::Physical(PhysicalSize::new(1600, 900)))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let mut render_ctx = RenderCtx::new(&window);

    let mut frame_count = 0;
    let mut frame_index = 0;

    let mut pressed_keys = HashSet::new();
    let mut running = true;

    let delta_time = 1.0 / 165.0;

    while running {
        event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent { event, window_id } => {
                    if window.id() == window_id {
                        match event {
                            WindowEvent::CloseRequested => running = false,
                            WindowEvent::KeyboardInput { input, .. } => {
                                if let Some(key_code) = input.virtual_keycode {
                                    if key_code == VirtualKeyCode::Escape {
                                        running = false;
                                    }

                                    match input.state {
                                        ElementState::Pressed => {
                                            if !pressed_keys.contains(&key_code) {
                                                pressed_keys.insert(key_code);
                                            }
                                        }
                                        ElementState::Released => {
                                            if pressed_keys.contains(&key_code) {
                                                pressed_keys.remove(&key_code);
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::MainEventsCleared => {
                    *control_flow = ControlFlow::Exit;
                }
                Event::DeviceEvent { event, .. } => {
                    match event {
                        DeviceEvent::MouseMotion { delta } => {
                            let camera_rig = &mut render_ctx.camera_rig;
                            camera_rig.driver_mut::<YawPitch>().rotate_yaw_pitch(0.3 * delta.0 as f32, 0.3 * delta.1 as f32);
                            camera_rig.update(delta_time);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        });

        let mut delta_pos = Vec3::ZERO;
        if pressed_keys.contains(&VirtualKeyCode::W) {
            delta_pos += Vec3::new(0.0, 0.0, 1.0);
        }
        if pressed_keys.contains(&VirtualKeyCode::A) {
            delta_pos += Vec3::new(-1.0, 0.0, 0.0);
        }
        if pressed_keys.contains(&VirtualKeyCode::S) {
            delta_pos += Vec3::new(0.0, 0.0, -1.0);
        }
        if pressed_keys.contains(&VirtualKeyCode::D) {
            delta_pos += Vec3::new(1.0, 0.0, 0.0);
        }
        delta_pos = render_ctx.camera_rig.final_transform.rotation * delta_pos;

        if pressed_keys.contains(&VirtualKeyCode::Space) {
            delta_pos += Vec3::new(0.0, 1.0, 0.0);
        }
        if pressed_keys.contains(&VirtualKeyCode::LShift) {
            delta_pos += Vec3::new(0.0, -1.0, 0.0);
        }

        let camera_rig = &mut render_ctx.camera_rig;
        camera_rig.driver_mut::<Position>().translate(-delta_pos * delta_time * 10.0);
        camera_rig.update(delta_time);

        renderer::render_frame(&render_ctx, &mut frame_index);

        frame_count += 1;
        frame_index = frame_count % render_ctx.frames.len();
    }
}
