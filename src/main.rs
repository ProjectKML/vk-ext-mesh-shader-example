use winit::{
    dpi::{PhysicalSize, Size},
    event::{Event, VirtualKeyCode, WindowEvent},
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

    let render_ctx = RenderCtx::new(&window);

    let mut frame_count = 0;
    let mut frame_index = 0;

    let mut running = true;

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
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::MainEventsCleared => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            }
        });

        renderer::render_frame(&render_ctx, &mut frame_index);

        frame_count += 1;
        frame_index = frame_count % render_ctx.frames.len();
    }
}
