use ash::vk;
use cgmath::Matrix4;
use crate::vk_call;
use super::*;
use super::color::LinearColor;

/// [`Frame`] implementation used by a [`VulkanRenderer`]
///
/// * See [`Renderer`] and [`Frame`]
#[derive(Debug)]
pub struct VulkanFrame<'a> {
    renderer: &'a VulkanRenderer,
    target: &'a VulkanTarget,
    setup: &'a RenderSetup,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    image_idx: u32,
    image_ready: [vk::Semaphore; 2],
    submit_ready: MaybeOwned<vk::Semaphore>,
    submit_fence: vk::Fence,
    output_size: Size<i32, Physical>,
    bound_pipeline: (vk::Pipeline, vk::PipelineLayout),
    transform_box: Matrix4<f32>,
}

impl<'a> Drop for VulkanFrame<'a> {
    fn drop(&mut self) {
        let device = self.renderer.device();
        unsafe {
            match &self.submit_ready {
                &MaybeOwned::Owned(v) if v != vk::Semaphore::null() => {
                    device.destroy_semaphore(v, None);
                },
                _ => {},
            }
            if !self.submit_fence.is_null() {
                device.destroy_fence(self.submit_fence, None);
            }
            self.image_ready.iter()
                .copied()
                .filter(|v| !v.is_null())
                .for_each(|semaphore| {
                    device.destroy_semaphore(semaphore, None);
                });
            if self.command_buffer != vk::CommandBuffer::null() {
                device.free_command_buffers(*self.renderer.command_pool, &[self.command_buffer]);
            }
        }
    }
}

impl<'a> VulkanFrame<'a> {
    #[inline(always)]
    fn frame_id(&self) -> usize {
        self.image_idx as _
    }
    fn bind_pipeline(&mut self, pipeline: vk::Pipeline, layout: vk::PipelineLayout) {
        if pipeline != self.bound_pipeline.0 {
            unsafe {
                self.renderer.device().cmd_bind_pipeline(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline
                );
            }
            self.bound_pipeline = (pipeline, layout);
        }
    }
    unsafe fn push_constants(&self, data: &UniformData) {
        use mem::offset_of;
        let device = self.renderer.device();
        device.push_constant(
            self.command_buffer,
            self.bound_pipeline.1,
            vk::ShaderStageFlags::VERTEX,
            0, &data.vert
        );
        const FRAG_OFFSET: usize = offset_of!(UniformData, frag);
        debug_assert_eq!(FRAG_OFFSET, 80);
        device.push_constant(
            self.command_buffer,
            self.bound_pipeline.1,
            vk::ShaderStageFlags::FRAGMENT,
            FRAG_OFFSET as _,
            &data.frag
        );
    }
    #[inline(always)]
    fn reset_viewport(&self) {
        let &Size { w, h, .. } = &self.output_size;
        unsafe {
            self.renderer.device().cmd_set_viewport(
                self.command_buffer, 0, &[vk::Viewport {
                    x: 0.0, y: 0.0,
                    width: w as _,
                    height: h as _,
                    min_depth: 0f32,
                    max_depth: 1f32,
                }]
            );
        }
    }

    #[inline(always)]
    unsafe fn reset_scissor(&self) {
        let &Size { w, h, .. } = &self.output_size;
        self.renderer.device().cmd_set_scissor(self.command_buffer, 0, &[
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: w as _,
                    height: h as _,
                },
            }
        ]);
    }

    unsafe fn rect_viewport<Coords>(&self, dst: &Rectangle<i32, Coords>) {
        self.renderer.device().cmd_set_viewport(
            self.command_buffer, 0, &[vk::Viewport {
                x: dst.loc.x as _,
                y: dst.loc.y as _,
                width: dst.size.w as _,
                height: dst.size.h as _,
                min_depth: 0f32,
                max_depth: 1f32,
            }]
        );
    }
}

impl<'a> Frame for VulkanFrame<'a> {
    type Error = <VulkanRenderer as Renderer>::Error;
    type TextureId = <VulkanRenderer as Renderer>::TextureId;

    fn id(&self) -> usize {
        self.renderer.id()
    }
    fn clear(&mut self, color: [f32; 4], at: &[Rectangle<i32, Physical>]) -> Result<(), Self::Error> {
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: LinearColor::from_srgb_premultiplied(&color).0,
            },
        };
        const DEFAULT_CLEAR_RECT: vk::ClearRect = vk::ClearRect {
            rect: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: 0, height: 0 },
            },
            base_array_layer: 0,
            layer_count: 1,
        };
        const fn clear_rect(rect: vk::Rect2D) -> vk::ClearRect {
            vk::ClearRect {
                rect,
                ..DEFAULT_CLEAR_RECT
            }
        }
        let cb = self.command_buffer;
        let device = self.renderer.device();
        let clear_attachments = [
            vk::ClearAttachment {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                color_attachment: 0,
                clear_value: clear_color,
            },
        ];
        let rect_it = at.iter().map(|r| vk::Rect2D {
            offset: vk::Offset2D { x: r.loc.x, y: r.loc.y },
            extent: vk::Extent2D {
                width: r.size.w as _,
                height: r.size.h as _,
            },
        });

        let mut clear_rects_static = [DEFAULT_CLEAR_RECT; 3];
        let clear_rects = if at.len() <= clear_rects_static.len() {
            rect_it.zip(&mut clear_rects_static).for_each(|(rect, out)| {
                out.rect = rect;
            });
            Cow::Borrowed(&clear_rects_static[..at.len()])
        } else {
            let mut rects = Vec::with_capacity(at.len());
            rects.extend(rect_it.map(clear_rect));
            Cow::Owned(rects)
        };
        unsafe {
            self.reset_viewport();
            device.cmd_clear_attachments(cb, &clear_attachments, &clear_rects);
        }
        Ok(())
    }

    #[tracing::instrument(skip(self, _damage))]
    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        color: [f32; 4],
    ) -> Result<(), Self::Error> {
        let (pipe, layout) = self.setup.quad_pipeline();
        self.bind_pipeline(pipe, layout.layout);
        let data = UniformData {
            vert: UniformDataVert {
                transform: self.transform_box,
                tex_offset: Vector2::new(0f32, 0f32),
                tex_extent: Vector2::new(1f32, 1f32),
            },
            frag: UniformDataFrag {
                color: LinearColor::from_srgb_premultiplied(&color).0,
            },
        };
        trace!(?data.vert, "pushing constants");
        unsafe {
            self.push_constants(&data);
            self.rect_viewport(&dst);
            self.renderer.device().cmd_draw(self.command_buffer, 4, 2, 0, 0);
        }
        Ok(())
    }
    #[tracing::instrument(skip(self, _damage, texture))]
    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        _damage: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        let device = self.renderer.device();
        let (pipe, layout) = self.setup.tex_pipeline();
        {
            let view = texture.0.get_or_create_view(layout)?;
            self.bind_pipeline(pipe, layout.layout);
            unsafe {
                let idx = self.frame_id();
                device.cmd_bind_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout.layout,
                    0,
                    &view.descriptor_sets()[idx..(idx+1)],
                    &[]
                );
            }
        }
        let (tw, th) = (texture.width(), texture.height());
        let data = UniformData {
            vert: UniformDataVert {
                transform: if let Some(src_transform) = src_transform.to_matrix() {
                    src_transform * self.transform_box
                } else {
                    self.transform_box
                },
                tex_offset: Vector2::new(
                    (src.loc.x / (tw as f64)) as f32,
                    (src.loc.y / (th as f64)) as f32,
                ),
                tex_extent: Vector2::new(
                    (src.size.w / (tw as f64)) as f32,
                    (src.size.h / (th as f64)) as f32,
                ),
            },
            frag: UniformDataFrag {
                alpha,
            },
        };
        trace!(?data.vert.transform, ?data.vert.tex_offset, ?data.vert.tex_extent, "pushing constants");
        unsafe {
            self.push_constants(&data);
            self.rect_viewport(&dst);
            self.renderer.device().cmd_draw(self.command_buffer, 4, 2, 0, 0);
        }
        Ok(())
    }
    fn transformation(&self) -> Transform {
        Transform::Normal
    }
    fn finish(mut self) -> Result<SyncPoint, Self::Error> {
        let cb = self.command_buffer;
        let device = self.renderer.device();

        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::empty())
            .build();
        let submit_fence = mem::take(&mut self.submit_fence);
        let submit_fence = if submit_fence != vk::Fence::null() {
            Ok(submit_fence)
        } else {
            unsafe {
                device.create_fence(&fence_info, None)
                    .vk("vkCreateFence")
            }
        }.map(|v| scopeguard::guard(v, |v| unsafe {
            device.destroy_fence(v, None);
        }))?;

        unsafe {
            device.cmd_end_render_pass(cb);
            device.end_command_buffer(cb)
        }.vk("vkEndCommandBuffer")?;
        unsafe {
            let acquire_stages =
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER;
            let signal_semaphores = [*self.submit_ready];
            let signal_semaphores: &[vk::Semaphore] = if signal_semaphores[0] == vk::Semaphore::null() {
                &[]
            } else {
                &signal_semaphores[..]
            };
            let wait_stages = [acquire_stages];
            let cmd_bufs = [self.command_buffer];
            let info = vk::SubmitInfo::builder()
                .wait_semaphores(&self.image_ready[..1])
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(signal_semaphores);
            device.queue_submit(
                self.renderer.queues.graphics.handle,
                &[info.build()],
                *submit_fence,
            ).vk("vkQueueSubmit")?;
            if let VulkanTarget::Surface(_, swapchain) = &self.target {
                let wait_sems = signal_semaphores;
                let swapchains = [swapchain.handle()];
                let image_indices = [
                    swapchain.images().iter().position(|swap_img| {
                        swap_img.image == self.image
                    }).map_or(0u32, |idx| idx as _)
                ];
                let mut results = [vk::Result::SUCCESS; 1];
                let info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(wait_sems)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices)
                    .results(&mut results);
                let _suboptimal = swapchain.extension().queue_present(
                    self.renderer.queues.graphics.handle,
                    &info
                ).vk("vkQueuePresentKHR")?;
                results.into_iter()
                    .find_map(|r| std::num::NonZeroI32::new(r.as_raw()))
                    .map_or(Ok(()), |v| Err(vk::Result::from_raw(v.get())))
                    .vk("vkQueuePresentKHR")?;
            }
        }

        let submitted = SubmittedFrame::from_frame(&mut self, ScopeGuard::into_inner(submit_fence));
        Ok(SyncPoint::from(VulkanFence::new(submitted, self.renderer.submitted_frames.0.clone())))
    }
    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        self.renderer.device.wait_fence_vk(sync)
    }
}

#[tracing::instrument(skip(r), name = "begin_frame")]
pub(super) fn render_internal(
    r: &mut VulkanRenderer,
    output_size: Size<i32, Physical>,
    _dst_transform: Transform,
) -> Result<VulkanFrame<'_>, <VulkanRenderer as Renderer>::Error> {
    let mut prev = r.cleanup();
    let target = r.target.as_ref().ok_or(Error::NoTarget)?;
    let new_command_buffer = || {
        r.device().create_single_command_buffer(*r.command_pool, vk::CommandBufferLevel::PRIMARY)
    };
    let command_buffer = if let Some(command_buffer) = prev.as_mut().map(|f| mem::take(&mut f.command_buffer)) {
        unsafe {
            r.device().reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES
            )
        }
            .vk("vkResetCommandBuffer")
            .map(|_| command_buffer)
            .or_else(|e| {
                warn!("error resetting command buffer: {}", &e);
                new_command_buffer()
            })
    } else {
        new_command_buffer()
    }.map(|v| scopeguard::guard(v, |v| unsafe {
        r.device().free_command_buffers(*r.command_pool, &[v]);
    }))?;

    let format = match target {
        VulkanTarget::Surface(_, swapchain) => swapchain.format(),
    };
    let setup = r.render_setups.get(&format)
        .expect("render setup did not exist");

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    vk_call!(r.device(), begin_command_buffer(*command_buffer, &begin_info))?;

    r.cmd_transition_images(*command_buffer, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let (
        image,
        image_idx,
        extent,
        framebuffer,
        image_ready,
        submit_ready,
    ) = match target {
        VulkanTarget::Surface(_, swapchain) => {
            let image_ready = prev.as_mut()
                .and_then(SubmittedFrame::take_image_ready)
                .map_or_else(|| unsafe {
                    let info = vk::SemaphoreCreateInfo::builder()
                        .flags(vk::SemaphoreCreateFlags::empty())
                        .build();
                    r.device().create_semaphore(&info, None)
                        .vk("vkCreateSemaphore")
                }, Ok)
                .map(|v| scopeguard::guard(v, |v| unsafe {
                    r.device().destroy_semaphore(v, None);
                }))?;
            let (image_idx, swap_image) = swapchain.acquire(u64::MAX, From::from(*image_ready))
                .or_else(|e| match e {
                    swapchain::AcquireError::Suboptimal(v) => {
                        warn!(?swapchain, "suboptimal swapchain");
                        Ok(v)
                    },
                    swapchain::AcquireError::Vk(e) => Err(Error::from(e)),
                })?;
            let (src_access, src_layout, src_stage) = if !swap_image.transitioned.fetch_or(true, Ordering::SeqCst) {
                unsafe {
                    let idx = image_idx as usize;
                    r.device().cmd_execute_commands(
                        *command_buffer,
                        &swapchain.init_buffers[idx..(idx+1)],
                    );
                }
                (vk::AccessFlags::TRANSFER_WRITE,
                 vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                 vk::PipelineStageFlags::TRANSFER)
            } else {
                (vk::AccessFlags::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags::TOP_OF_PIPE)
            };
            // ret.image = swap_image.image;
            // ret.image_idx = image_idx;
            // ret.submit_ready = MaybeOwned::Borrowed(swap_image.submit_semaphore);
            let acquire_access =
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_READ;
            let acquire_barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(src_access)
                .dst_access_mask(acquire_access)
                .old_layout(src_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swap_image.image)
                .subresource_range(COLOR_SINGLE_LAYER);
            unsafe {
                r.device().cmd_pipeline_barrier(
                    *command_buffer,
                    src_stage,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[], &[acquire_barrier.build()]
                );
            }
            (
                swap_image.image,
                image_idx,
                *swapchain.extent(),
                swap_image.framebuffer,
                ScopeGuard::into_inner(image_ready),
                swap_image.submit_semaphore,
            )
        },
    };
    let submit_ready = if submit_ready != vk::Semaphore::null() {
        MaybeOwned::Borrowed(submit_ready)
    } else {
        MaybeOwned::Owned(vk::Semaphore::null())
    };
    let submit_fence = prev.as_mut()
        .and_then(SubmittedFrame::take_fence)
        .unwrap_or_default();

    let ret = VulkanFrame {
        renderer: r,
        target,
        setup,
        command_buffer: ScopeGuard::into_inner(command_buffer),
        image,
        image_idx,
        image_ready: [
            image_ready,
            vk::Semaphore::null(),
        ],
        submit_ready,
        submit_fence,
        output_size,
        bound_pipeline: (vk::Pipeline::null(), vk::PipelineLayout::null()),
        transform_box: transform::MAT4_MODEL_BOX,
    };
    let cb = ret.command_buffer;
    let begin_info = vk::RenderPassBeginInfo::builder()
        .render_pass(setup.render_pass())
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        });
    debug_assert_eq!(ret.output_size.w as u32, extent.width);
    debug_assert_eq!(ret.output_size.h as u32, extent.height);
    unsafe {
        r.device().cmd_begin_render_pass(cb, &begin_info, vk::SubpassContents::INLINE);
        ret.reset_viewport();
        ret.reset_scissor();
    }
    Ok(ret)
}

/// Contains handles that need to live on past [`Frame::finish`]
#[derive(Debug)]
pub(super) struct SubmittedFrame {
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    image_ready: Option<NonZeroU64>,
    fence: vk::Fence,
    device: Arc<Device>,
}

impl SubmittedFrame {
    #[inline(always)]
    fn from_frame(frame: &mut VulkanFrame<'_>, fence: vk::Fence) -> Self {
        use vk::Handle;
        let image_ready = mem::take(&mut frame.image_ready[0]);
        SubmittedFrame {
            command_buffer: mem::take(&mut frame.command_buffer),
            command_pool: *frame.renderer.command_pool,
            image_ready: NonZeroU64::new(image_ready.as_raw()),
            fence,
            device: frame.renderer.device.clone(),
        }
    }

    #[inline(always)]
    pub fn status(&self) -> Result<FenceStatus, vk::Result> {
        self.device.fence_status(self.fence)
    }

    pub fn is_done(&self) -> bool {
        match self.status() {
            Ok(FenceStatus::Signaled) => true,
            Ok(FenceStatus::Unsignaled) => false,
            Err(error) => {
                error!(?error, "error getting fence status");
                false
            },
        }
    }

    #[inline]
    pub fn wait(&self, timeout: u64) -> Result<(), vk::Result> {
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, timeout)
        }
    }

    #[inline(always)]
    fn take_image_ready(&mut self) -> Option<vk::Semaphore> {
        use vk::Handle;
        self.image_ready.take()
            .map(|v| vk::Semaphore::from_raw(v.get()))
    }

    fn take_fence(&mut self) -> Option<vk::Fence> {
        Some(mem::take(&mut self.fence))
            .filter(|v| !v.is_null())
            .and_then(|fence| unsafe {
                match self.device.reset_fences(&[fence]).vk("vkResetFences") {
                    Ok(..) => Some(fence),
                    Err(error) => {
                        error!(?error, "error resetting fence");
                        self.fence = fence;
                        None
                    },
                }
            })
    }

    #[inline(always)]
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for SubmittedFrame {
    fn drop(&mut self) {
        use vk::Handle;
        unsafe {
            if !self.command_buffer.is_null() && !self.command_pool.is_null() {
                self.device.free_command_buffers(self.command_pool, &[self.command_buffer]);
            }
            if let Some(v) = self.image_ready {
                let v = vk::Semaphore::from_raw(v.get());
                self.device.destroy_semaphore(v, None);
            }
            if !self.fence.is_null() {
                self.device.destroy_fence(self.fence, None);
            }
        }
    }
}

/// Simple wrapper for keeping track of whether or not a given raw vulkan handle is owned or just a
/// borrowed [`Copy`] variant
#[derive(Debug, Clone, Copy)]
enum MaybeOwned<T> {
    Borrowed(T),
    Owned(T),
}

impl<T> Deref for MaybeOwned<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            MaybeOwned::Borrowed(v) => v,
            MaybeOwned::Owned(v) => v,
        }
    }
}
