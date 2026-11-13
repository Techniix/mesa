/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nvrm.h"
#include "nvkmd_nvrm_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <unistd.h>

#include "vk_log.h"

#include "util/u_memory.h"
#include "util/u_dynarray.h"
#include "util/macros.h"
#include "nv_push_clc36f.h"

#include "class/cl0002.h" // NV01_CONTEXT_DMA
#include "class/clc361.h" // NVC361_NOTIFY_CHANNEL_PENDING
#include "class/clc46f.h" // TURING_CHANNEL_GPFIFO_A
#include "class/cl2080_notification.h" // NV2080_ENGINE_TYPE_GRAPHICS
#include "class/clc36f.h" // VOLTA_CHANNEL_GPFIFO_A
#include "class/cla16f.h" // KeplerBControlGPFifo
#include "class/cl906fsw.h" // NV906F_NOTIFIERS_RC
#include "class/cl0005.h" // NV01_EVENT
#include "class/cl00fe.h" // NV_MEMORY_MAPPER
#include "ctrl/ctrla06f/ctrla06fgpfifo.h" // NVA06F_CTRL_CMD_BIND
#include "ctrl/ctrlc36f.h" // NVC36F_CTRL_CMD_INTERNAL_GPFIFO_GET_WORK_SUBMIT_TOKEN
#include "ctrl/ctrl00fe.h" // NV00FE_CTRL_CMD_SUBMIT_OPERATIONS

#include "nvRmSemSurf.h"

#define SUBC_NVC36F 0


/* Logs the failing call verbatim with its NV_STATUS and unwinds to `error:`. */
#define NV_CHECK(call) NVRM_CHECK(call)
#define VK_CHECK(vkResIn) {VkResult _vkRes = vkResIn; if (_vkRes != VK_SUCCESS) {vkRes = vk_error(log_obj, _vkRes); goto error;}}

#define NVRM_CTX_CMDBUF_BYTES   0x80000
#define NVRM_CTX_GPFIFO_ENTRIES 0x8000

/* Bind-channel <-> NV_MEMORY_MAPPER rendezvous slots in the bind ctx semsurf. */
#define NVRM_BIND_SLOT_GO   0
#define NVRM_BIND_SLOT_DONE 1
/* Mapper kernel-queue depth; each rendezvous chunk also spends a WAIT + SIGNAL
 * op, so a chunk carries at most this minus two paging ops. */
#define NVRM_BIND_MAPPER_QUEUE_SIZE 256

extern const struct nvkmd_ctx_ops nvkmd_nvrm_exec_ctx_ops;
extern const struct nvkmd_ctx_ops nvkmd_nvrm_bind_ctx_ops;


/*
 * Shared GPFIFO channel
 */

/* Completed-fence value: the highest wSeq the GPU has released into chan->sem. */
static uint64_t
chan_completed(struct nvkmd_nvrm_channel *chan)
{
	return *(volatile uint64_t *)chan->sem->map;
}

/* True once RM has reported a robust-channel error on this channel. */
static bool
chan_check_error(struct nvkmd_nvrm_channel *chan)
{
	volatile NvNotification *notifiers = chan->notifier->map;
	return notifiers[NV906F_NOTIFIERS_RC].status != 0;
}

static uint32_t
chan_slot_dwords(void)
{
	return (NVRM_CTX_CMDBUF_BYTES / 4) / NVRM_CTX_CMDBUF_SLOTS;
}

/* GPU virtual address of the current cmdBuf slot. */
static uint64_t
chan_slot_gpu_base(struct nvkmd_nvrm_channel *chan)
{
	return chan->cmdBuf->va->addr +
		(uint64_t)chan->cmdBufCurSlot * chan_slot_dwords() * 4;
}

/* (Re)initialise the push writer into the current cmdBuf slot. */
static void
chan_begin_slot(struct nvkmd_nvrm_channel *chan)
{
	uint32_t slotDwords = chan_slot_dwords();
	uint32_t *base = (uint32_t *)chan->cmdBuf->map + chan->cmdBufCurSlot * slotDwords;
	nv_push_init(&chan->push, base, slotDwords, BITFIELD_BIT(SUBC_NV9097));
	chan->cmdBufSubmittedDw = 0;
}

static void
chan_write_gp_fifo_entry(struct nvkmd_nvrm_channel *chan, const struct nvkmd_ctx_exec *exec)
{
	uint32_t next = (uint32_t)((chan->gpPut + 1) % NVRM_CTX_GPFIFO_ENTRIES);

	/* Back-pressure: never let GPPut catch the host's GPGet. Earlier
	 * submissions have been doorbelled, so the host keeps draining and GPGet
	 * advances; bail out on a dead channel rather than spin forever. */
	KeplerBControlGPFifo *userD = chan->userD->map;
	while (next == (userD->GPGet % NVRM_CTX_GPFIFO_ENTRIES)) {
		if (chan_check_error(chan))
			break;
		struct pollfd pfd = { .fd = chan->osEvent, .events = POLLIN | POLLPRI };
		poll(&pfd, 1, 100);
	}

	uint32_t *ptr = (uint32_t*)chan->gpFifo->map + 2*chan->gpPut;

	ptr[0] = DRF_NUM(A16F, _GP_ENTRY0, _GET, NvU64_LO32(exec->addr) >> 2);
	ptr[1] =
		DRF_NUM(A16F, _GP_ENTRY1, _GET_HI, NvU64_HI32(exec->addr)) |
		DRF_NUM(A16F, _GP_ENTRY1, _LENGTH, (exec->size_B >> 2)) |
		DRF_NUM(A16F, _GP_ENTRY1, _SYNC, (exec->no_prefetch ? NVA16F_GP_ENTRY1_SYNC_WAIT : NVA16F_GP_ENTRY1_SYNC_PROCEED));

	chan->gpPut = next;
}

/* Turn the dwords written to chan->push since the last segment into one GP
 * entry. Lets wait() (acquires) and signal()/flush() (releases + fence) land
 * as separate GP entries on either side of the exec entries. */
static void
chan_emit_push_segment(struct nvkmd_nvrm_channel *chan)
{
	uint32_t total = nv_push_dw_count(&chan->push);
	if (total <= chan->cmdBufSubmittedDw)
		return;

	struct nvkmd_ctx_exec exec = {
		.addr = chan_slot_gpu_base(chan) + (uint64_t)chan->cmdBufSubmittedDw * 4,
		.size_B = (total - chan->cmdBufSubmittedDw) * 4,
	};
	chan_write_gp_fifo_entry(chan, &exec);
	chan->cmdBufSubmittedDw = total;
}

static void
write_semaphore_release(struct nv_push *push, uint64_t adrGpu, uint64_t value, bool waitForIdle)
{
   P_MTHD(push, NVC36F, SEM_ADDR_LO);
   P_NVC36F_SEM_ADDR_LO(push, (uint32_t)adrGpu >> 2);
   P_NVC36F_SEM_ADDR_HI(push, (uint32_t)(adrGpu >> 32));
   P_NVC36F_SEM_PAYLOAD_LO(push, (uint32_t)value);
   P_NVC36F_SEM_PAYLOAD_HI(push, (uint32_t)(value >> 32));
   P_NVC36F_SEM_EXECUTE(push, {
      .operation = OPERATION_RELEASE,
      .release_wfi = waitForIdle ? RELEASE_WFI_EN : RELEASE_WFI_DIS,
      .payload_size = PAYLOAD_SIZE_64BIT,
      .release_timestamp = RELEASE_TIMESTAMP_DIS,
   });
   P_MTHD(push, NVC36F, NON_STALL_INTERRUPT);
   P_NVC36F_NON_STALL_INTERRUPT(push, 0);
}

static void
write_semaphore_acquire(struct nv_push *push, uint64_t adrGpu, uint64_t value)
{
   P_MTHD(push, NVC36F, SEM_ADDR_LO);
   P_NVC36F_SEM_ADDR_LO(push, (uint32_t)adrGpu >> 2);
   P_NVC36F_SEM_ADDR_HI(push, (uint32_t)(adrGpu >> 32));
   P_NVC36F_SEM_PAYLOAD_LO(push, (uint32_t)value);
   P_NVC36F_SEM_PAYLOAD_HI(push, (uint32_t)(value >> 32));
   P_NVC36F_SEM_EXECUTE(push, {
      .operation = OPERATION_ACQ_STRICT_GEQ,
      .acquire_switch_tsg = ACQUIRE_SWITCH_TSG_EN,
      .payload_size = PAYLOAD_SIZE_64BIT,
   });
}

static void
nvrm_channel_finish(struct nvkmd_nvrm_dev *dev, struct nvkmd_nvrm_channel *chan)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   if (chan->hEvent != 0)
      nvRmApiFree(&rm, chan->hEvent);
   if (chan->osEvent >= 0)
      close(chan->osEvent);
   if (chan->hChannel != 0)
      nvRmApiFree(&rm, chan->hChannel);
   if (chan->hCtxDma != 0)
      nvRmApiFree(&rm, chan->hCtxDma);
   if (chan->sem != NULL)
      nvkmd_mem_unref(chan->sem);
   if (chan->cmdBuf != NULL)
      nvkmd_mem_unref(chan->cmdBuf);
   if (chan->gpFifo != NULL)
      nvkmd_mem_unref(chan->gpFifo);
   if (chan->userD != NULL)
      nvkmd_mem_unref(chan->userD);
   if (chan->notifier != NULL)
      nvkmd_mem_unref(chan->notifier);
}

/* Allocate and schedule a bare GPFIFO channel: the memories, context DMA,
 * channel, work-submit token and OS wake event, ready for host SEM methods.
 * Engine subchannel objects (if any) are the caller's responsibility. */
static VkResult
nvrm_channel_init(struct nvkmd_nvrm_dev *dev,
                  struct vk_object_base *log_obj,
                  struct nvkmd_nvrm_channel *chan)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct nvkmd_dev *_dev = &dev->base;
   VkResult vkRes;

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   chan->osEvent = -1;

   VK_CHECK(nvkmd_dev_alloc_mapped_mem(_dev, log_obj,  0x1000,  0x1000, NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR,  &chan->notifier));
   VK_CHECK(nvkmd_dev_alloc_mapped_mem(_dev, log_obj, 0x80000, 0x10000, NVKMD_MEM_LOCAL, NVKMD_MEM_MAP_RDWR, &chan->userD));
   VK_CHECK(nvkmd_dev_alloc_mapped_mem(_dev, log_obj, 0x40000,  0x1000, NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR,  &chan->gpFifo));
   VK_CHECK(nvkmd_dev_alloc_mapped_mem(_dev, log_obj, NVRM_CTX_CMDBUF_BYTES, 0x1000, NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR, &chan->cmdBuf));
   VK_CHECK(nvkmd_dev_alloc_mapped_mem(_dev, log_obj,  0x1000,  0x1000, NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR,  &chan->sem));

   /* The fence value and the RC error notifier are read in steady state
    * (chan_completed / chan_check_error); start them at zero rather than
    * trusting freshly allocated GART to be cleared. */
   memset(chan->sem->map, 0, chan->sem->size_B);
   memset(chan->notifier->map, 0, chan->notifier->size_B);

	NV_CONTEXT_DMA_ALLOCATION_PARAMS ctxDmaParams = {
		.flags =
			DRF_DEF(OS03, _FLAGS, _MAPPING, _KERNEL) |
			DRF_DEF(OS03, _FLAGS, _HASH_TABLE, _DISABLE),

		.hMemory = nvkmd_nvrm_mem(chan->notifier)->hMemoryPhys,
		.offset = 0,
		.limit = chan->notifier->size_B - 1,
	};
   NV_CHECK(nvRmApiAlloc(&rm, pdev->hDevice, &chan->hCtxDma, NV01_CONTEXT_DMA, &ctxDmaParams));

   NvU32 engineType = NV2080_ENGINE_TYPE_GRAPHICS;
	NV_CHANNEL_ALLOC_PARAMS createChannelParams = {
		.hObjectError  = chan->hCtxDma,
		.gpFifoOffset  = chan->gpFifo->va->addr,
		.gpFifoEntries = 0x8000,
		.flags         = 0,
		.hVASpace      = pdev->hVaSpace,
		.hUserdMemory  = {nvkmd_nvrm_mem(chan->userD)->hMemoryPhys},
		.userdOffset   = {0},
		.engineType    = engineType,
	};
   NV_CHECK(nvRmApiAlloc(&rm, pdev->hDevice, &chan->hChannel, pdev->channelClass, &createChannelParams));

	NVA06F_CTRL_BIND_PARAMS bindParams = {.engineType = engineType};
	NV_CHECK(nvRmApiControl(&rm, chan->hChannel, NVA06F_CTRL_CMD_BIND, &bindParams, sizeof(bindParams)));

	NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS scheduleParams = {.bEnable = NV_TRUE};
	NV_CHECK(nvRmApiControl(&rm, chan->hChannel, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, &scheduleParams, sizeof(scheduleParams)));

	NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS notifParams = {
		.index = NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN
	};
	NV_CHECK(nvRmApiControl(&rm,
		chan->hChannel,
		NVC36F_CTRL_CMD_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX,
		&notifParams,
		sizeof(notifParams)
	));

	NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS tokenParams = {0};
	NV_CHECK(nvRmApiControl(&rm,
		chan->hChannel,
		NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
		&tokenParams,
		sizeof(tokenParams)
	));

   chan->osEvent = open(rm.nodeName, O_RDWR | O_CLOEXEC);
   if (chan->osEvent < 0) {
      vkRes = vk_errorf(log_obj, VK_ERROR_INITIALIZATION_FAILED,
                        "open(%s) failed: %s", rm.nodeName, strerror(errno));
      goto error;
   }
   struct NvRmApi rmOsEvent = rm;
   rmOsEvent.fd = chan->osEvent;
   NV_CHECK(nvRmApiAllocOsEvent(&rmOsEvent, chan->osEvent));

	NV0005_ALLOC_PARAMETERS eventParams = {
		.hParentClient = pdev->hClient,
		.hSrcResource = pdev->hSubdevice,
		.hClass = NV01_EVENT_OS_EVENT,
		.notifyIndex =
			NV2080_NOTIFIERS_FIFO_EVENT_MTHD |
			NV01_EVENT_NONSTALL_INTR |
			NV01_EVENT_WITHOUT_EVENT_DATA |
			NV01_EVENT_SUBDEVICE_SPECIFIC |
			DRF_NUM(0005, _NOTIFY_INDEX, _SUBDEVICE, 0),
		.data = (NvP64)(uintptr_t)chan->osEvent,
	};
   NV_CHECK(nvRmApiAlloc(&rm, pdev->hSubdevice, &chan->hEvent, NV01_EVENT_OS_EVENT, &eventParams));

   chan_begin_slot(chan);

   return VK_SUCCESS;

error:
   /* Leave the partially-initialised channel for the caller's ctx destroy to
    * tear down via nvrm_channel_finish (which NULL/zero-guards each field), so
    * it isn't freed twice. */
   return vkRes;
}

/* Seal the channel fence into a final GP entry, ring the doorbell, recycle the
 * cmdBuf slot ring (stalling only if the pipeline is full), and reset the push
 * writer. Non-blocking otherwise: the work is submitted, not awaited. */
static VkResult
nvrm_channel_submit(struct nvkmd_nvrm_dev *dev,
                    struct nvkmd_nvrm_channel *chan,
                    struct vk_object_base *log_obj)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   NvNotification *notifiers = chan->notifier->map;
   NvNotification *submitTokenNotifier = &notifiers[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN];
   KeplerBControlGPFifo *userD = chan->userD->map;
   uint64_t semAdrGpu = chan->sem->va->addr;

   chan->wSeq++;

   /* Channel fence: the drained-state marker the slot-recycle wait polls on. */
   write_semaphore_release(&chan->push, semAdrGpu, chan->wSeq, true);
   chan_emit_push_segment(chan);

   userD->GPPut = chan->gpPut;

   volatile NvU32 *doorbell = (void*)((NvU8*)pdev->usermodeMap.address + NVC361_NOTIFY_CHANNEL_PENDING);
   *doorbell = submitTokenNotifier->info32;

   chan->cmdBufSlotFence[chan->cmdBufCurSlot] = chan->wSeq;
   chan->cmdBufCurSlot = (chan->cmdBufCurSlot + 1) % NVRM_CTX_CMDBUF_SLOTS;

   VkResult result = VK_SUCCESS;
   while (chan_completed(chan) < chan->cmdBufSlotFence[chan->cmdBufCurSlot]) {
      if (chan_check_error(chan)) {
         result = vk_error(log_obj, VK_ERROR_DEVICE_LOST);
         break;
      }
      struct pollfd pfd = { .fd = chan->osEvent, .events = POLLIN | POLLPRI };
      poll(&pfd, 1, 1000);
   }

   chan_begin_slot(chan);

   return result;
}

/* Block until every submission on the channel has retired. */
static VkResult
nvrm_channel_sync(struct nvkmd_nvrm_channel *chan, struct vk_object_base *log_obj)
{
   while (chan_completed(chan) < chan->wSeq) {
      if (chan_check_error(chan))
         return vk_error(log_obj, VK_ERROR_DEVICE_LOST);
      struct pollfd pfd = { .fd = chan->osEvent, .events = POLLIN | POLLPRI };
      poll(&pfd, 1, 1000);
   }
   return VK_SUCCESS;
}


/*
 * Exec context
 */

static VkResult
nvkmd_nvrm_create_exec_ctx(struct nvkmd_dev *_dev,
                              struct vk_object_base *log_obj,
                              enum nvkmd_engines engines,
                              struct nvkmd_ctx **ctx_out)
{
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   VkResult vkRes;

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   struct nvkmd_nvrm_exec_ctx *ctx = CALLOC_STRUCT(nvkmd_nvrm_exec_ctx);
   if (ctx == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   ctx->base.ops = &nvkmd_nvrm_exec_ctx_ops;
   ctx->base.dev = &dev->base;

   VK_CHECK(nvrm_channel_init(dev, log_obj, &ctx->chan));

   NV_CHECK(nvRmApiAlloc(&rm, ctx->chan.hChannel, &ctx->subchannels.hCopy, pdev->base.dev_info.cls_copy, NULL));
   NV_CHECK(nvRmApiAlloc(&rm, ctx->chan.hChannel, &ctx->subchannels.hEng2d, pdev->base.dev_info.cls_eng2d, NULL));
   NV_CHECK(nvRmApiAlloc(&rm, ctx->chan.hChannel, &ctx->subchannels.hEng3d, pdev->base.dev_info.cls_eng3d, NULL));
   NV_CHECK(nvRmApiAlloc(&rm, ctx->chan.hChannel, &ctx->subchannels.hM2mf, pdev->base.dev_info.cls_m2mf, NULL));
   NV_CHECK(nvRmApiAlloc(&rm, ctx->chan.hChannel, &ctx->subchannels.hCompute, pdev->base.dev_info.cls_compute, NULL));

   *ctx_out = &ctx->base;
   return VK_SUCCESS;

error:
   nvkmd_ctx_destroy(&ctx->base);
	return vkRes;
}

static void
nvkmd_nvrm_exec_ctx_destroy(struct nvkmd_ctx *_ctx)
{
   struct nvkmd_nvrm_exec_ctx *ctx = nvkmd_nvrm_exec_ctx(_ctx);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(ctx->base.dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   if (ctx->subchannels.hCopy != 0)
      nvRmApiFree(&rm, ctx->subchannels.hCopy);
   if (ctx->subchannels.hEng2d != 0)
      nvRmApiFree(&rm, ctx->subchannels.hEng2d);
   if (ctx->subchannels.hEng3d != 0)
      nvRmApiFree(&rm, ctx->subchannels.hEng3d);
   if (ctx->subchannels.hM2mf != 0)
      nvRmApiFree(&rm, ctx->subchannels.hM2mf);
   if (ctx->subchannels.hCompute != 0)
      nvRmApiFree(&rm, ctx->subchannels.hCompute);

   nvrm_channel_finish(dev, &ctx->chan);

   FREE(ctx);
}

static VkResult
nvkmd_nvrm_exec_ctx_wait(struct nvkmd_ctx *_ctx,
                            struct vk_object_base *log_obj,
                            uint32_t wait_count,
                            const struct vk_sync_wait *waits)
{
   struct nvkmd_nvrm_exec_ctx *ctx = nvkmd_nvrm_exec_ctx(_ctx);

   /* In-channel semaphore acquire per wait. They must precede the exec GP
    * entries, so seal them into their own GP entry now. Safe now that flush()
    * is non-blocking: the host stalls on the acquire without the CPU holding
    * a fence wait, so wait-before-signal can't deadlock. */
   for (uint32_t i = 0; i < wait_count; i++) {
      uint64_t addr = nvkmd_nvrm_sync_gpu_addr(waits[i].sync);
      uint64_t value = nvkmd_nvrm_sync_gpu_wait_value(waits[i].sync, waits[i].wait_value);
      write_semaphore_acquire(&ctx->chan.push, addr, value);
   }
   chan_emit_push_segment(&ctx->chan);

   return VK_SUCCESS;
}

static VkResult
nvkmd_nvrm_exec_ctx_flush(struct nvkmd_ctx *_ctx,
                             struct vk_object_base *log_obj)
{
   struct nvkmd_nvrm_exec_ctx *ctx = nvkmd_nvrm_exec_ctx(_ctx);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(ctx->base.dev);

   return nvrm_channel_submit(dev, &ctx->chan, log_obj);
}

static VkResult
nvkmd_nvrm_exec_ctx_exec(struct nvkmd_ctx *_ctx,
                            struct vk_object_base *log_obj,
                            uint32_t exec_count,
                            const struct nvkmd_ctx_exec *execs)
{
   struct nvkmd_nvrm_exec_ctx *ctx = nvkmd_nvrm_exec_ctx(_ctx);

   for (uint32_t i = 0; i < exec_count; i++) {
      chan_write_gp_fifo_entry(&ctx->chan, &execs[i]);
   }

   return VK_SUCCESS;
}

static VkResult
nvkmd_nvrm_exec_ctx_signal(struct nvkmd_ctx *_ctx,
                              struct vk_object_base *log_obj,
                              uint32_t signal_count,
                              const struct vk_sync_signal *signals)
{
   struct nvkmd_nvrm_exec_ctx *ctx = nvkmd_nvrm_exec_ctx(_ctx);

   /* Emit an in-channel semaphore release per signal, after the exec work.
    * These stay in ctx->chan.push and are sealed into a GP entry by flush()
    * (alongside the channel fence). */
   for (uint32_t i = 0; i < signal_count; i++) {
      uint64_t addr = nvkmd_nvrm_sync_gpu_addr(signals[i].sync);
      uint64_t value = nvkmd_nvrm_sync_gpu_signal_value(signals[i].sync, signals[i].signal_value);
      write_semaphore_release(&ctx->chan.push, addr, value, true);
   }

   return nvkmd_nvrm_exec_ctx_flush(&ctx->base, log_obj);
}

static VkResult
nvkmd_nvrm_exec_ctx_sync(struct nvkmd_ctx *_ctx,
                            struct vk_object_base *log_obj)
{
   struct nvkmd_nvrm_exec_ctx *ctx = nvkmd_nvrm_exec_ctx(_ctx);

   VkResult result = nvkmd_nvrm_exec_ctx_flush(&ctx->base, log_obj);
   if (result != VK_SUCCESS)
      return result;

   /* The explicit idle point: block until everything submitted has retired. */
   return nvrm_channel_sync(&ctx->chan, log_obj);
}

const struct nvkmd_ctx_ops nvkmd_nvrm_exec_ctx_ops = {
   .destroy = nvkmd_nvrm_exec_ctx_destroy,
   .wait = nvkmd_nvrm_exec_ctx_wait,
   .exec = nvkmd_nvrm_exec_ctx_exec,
   .signal = nvkmd_nvrm_exec_ctx_signal,
   .flush = nvkmd_nvrm_exec_ctx_flush,
   .sync = nvkmd_nvrm_exec_ctx_sync,
};


/*
 * Bind context
 *
 * Sparse (un)binding runs on NV_MEMORY_MAPPER, which does the MAP/UNMAP from a
 * kernel worker and synchronizes only through a semaphore surface. The bind
 * channel bridges that to the event-object vk_syncs entirely on the GPU
 * timeline, so flush() never blocks the CPU: it acquires the wait vk_syncs,
 * releases a GO slot (+ non-stall interrupt) the mapper waits on, then acquires
 * the DONE slot the mapper signals and releases the signal vk_syncs. The semsurf
 * is the one place this backend uses NV_SEMAPHORE_SURFACE.
 */

static uint64_t
bind_slot_gpu_addr(struct nvkmd_nvrm_bind_ctx *ctx, uint32_t slot)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(ctx->base.dev->pdev);
   return ctx->semSurf->memory->va->addr + (uint64_t)slot * pdev->semSurfLayout.size;
}

static VkResult
nvkmd_nvrm_create_bind_ctx(struct nvkmd_dev *_dev,
                              struct vk_object_base *log_obj,
                              struct nvkmd_ctx **ctx_out)
{
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   VkResult vkRes;

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   struct nvkmd_nvrm_bind_ctx *ctx = CALLOC_STRUCT(nvkmd_nvrm_bind_ctx);
   if (ctx == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   ctx->base.ops = &nvkmd_nvrm_bind_ctx_ops;
   ctx->base.dev = &dev->base;
   ctx->mapperQueueSize = NVRM_BIND_MAPPER_QUEUE_SIZE;
   util_dynarray_init(&ctx->waits, NULL);
   util_dynarray_init(&ctx->signals, NULL);
   util_dynarray_init(&ctx->ops, NULL);

   VK_CHECK(nvrm_channel_init(dev, log_obj, &ctx->chan));

   /* The mapper and the channel rendezvous through this semsurf. 64-bit
    * semaphores are assumed (Turing+), so the channel's 64-bit SEM methods
    * match RM's view of each slot. */
   assert(pdev->semSurfLayout.caps &
          NV2080_CTRL_FB_GET_SEMAPHORE_SURFACE_LAYOUT_CAPS_64BIT_SEMAPHORES_SUPPORTED);
   NV_STATUS nvRes = nvRmSemSurfCreate(dev, 0x1000, &ctx->semSurf);
   if (nvRes != NV_OK) {
      vkRes = vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                        "nvRmSemSurfCreate failed: 0x%08x (%s)",
                        (unsigned)nvRes, nvkmd_nvrm_status_str(nvRes));
      goto error;
   }

   /* Bind the semsurf to the bind channel so RM rescans it on the channel's
    * non-stall interrupt and wakes the mapper's GO waiter. */
   NvU32 notify = NV2080_NOTIFIERS_FIFO_EVENT_MTHD;
   NV_CHECK(nvRmSemSurfBindChannel(ctx->semSurf, ctx->chan.hChannel, 1, &notify));

   VK_CHECK(nvkmd_dev_alloc_mapped_mem(_dev, log_obj, 0x1000, 0x1000,
                                       NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR,
                                       &ctx->notifyMem));
   memset(ctx->notifyMem->map, 0, ctx->notifyMem->size_B);

   NV_MEMORY_MAPPER_ALLOCATION_PARAMS mmParams = {
      .hSemaphoreSurface   = ctx->semSurf->hSemSurf,
      .maxQueueSize        = ctx->mapperQueueSize,
      .hNotificationMemory = nvkmd_nvrm_mem(ctx->notifyMem)->hMemoryPhys,
      .notificationOffset  = 0,
   };
   NV_CHECK(nvRmApiAlloc(&rm, pdev->hSubdevice, &ctx->hMemMapper, NV_MEMORY_MAPPER, &mmParams));

   *ctx_out = &ctx->base;
   return VK_SUCCESS;

error:
   nvkmd_ctx_destroy(&ctx->base);
   return vkRes;
}

static void
nvkmd_nvrm_bind_ctx_destroy(struct nvkmd_ctx *_ctx)
{
   struct nvkmd_nvrm_bind_ctx *ctx = nvkmd_nvrm_bind_ctx(_ctx);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(ctx->base.dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   if (ctx->hMemMapper != 0)
      nvRmApiFree(&rm, ctx->hMemMapper);
   if (ctx->semSurf != NULL) {
      if (ctx->chan.hChannel != 0) {
         NvU32 notify = NV2080_NOTIFIERS_FIFO_EVENT_MTHD;
         nvRmSemSurfUnbindChannel(ctx->semSurf, ctx->chan.hChannel, 1, &notify);
      }
      nvRmSemSurfDestroy(ctx->semSurf);
   }
   if (ctx->notifyMem != NULL)
      nvkmd_mem_unref(ctx->notifyMem);

   nvrm_channel_finish(dev, &ctx->chan);

   util_dynarray_fini(&ctx->waits);
   util_dynarray_fini(&ctx->signals);
   util_dynarray_fini(&ctx->ops);

   FREE(ctx);
}

static VkResult
nvkmd_nvrm_bind_ctx_wait(struct nvkmd_ctx *_ctx,
                            struct vk_object_base *log_obj,
                            uint32_t wait_count,
                            const struct vk_sync_wait *waits)
{
   struct nvkmd_nvrm_bind_ctx *ctx = nvkmd_nvrm_bind_ctx(_ctx);

   for (uint32_t i = 0; i < wait_count; i++) {
      struct nvkmd_nvrm_bind_sem s = {
         .addr = nvkmd_nvrm_sync_gpu_addr(waits[i].sync),
         .value = nvkmd_nvrm_sync_gpu_wait_value(waits[i].sync, waits[i].wait_value),
      };
      util_dynarray_append_typed(&ctx->waits, struct nvkmd_nvrm_bind_sem, s);
   }

   return VK_SUCCESS;
}

static int
va_range_cmp(const void *a, const void *b)
{
   const struct nvkmd_nvrm_va_range *ra = a, *rb = b;
   if (ra->start < rb->start) return -1;
   if (ra->start > rb->start) return 1;
   return 0;
}

/* Queue an UNMAP op for each currently-mapped sub-range of [off, end) on `va`.
 * NV_MEMORY_MAPPER's MAP rejects an already-mapped range, so a rebind must
 * first clear exactly the live overlap (and never touch unmapped pages, which
 * UNMAP would reject). */
static void
va_emit_unmap_overlaps(struct nvkmd_nvrm_bind_ctx *ctx,
                       struct nvkmd_nvrm_va *va, uint64_t off, uint64_t end)
{
   util_dynarray_foreach(&va->bound, struct nvkmd_nvrm_va_range, r) {
      uint64_t lo = MAX2(r->start, off);
      uint64_t hi = MIN2(r->end, end);
      if (lo < hi) {
         struct nvkmd_nvrm_bind_op op = {
            .unmap          = true,
            .hVirtualMemory = va->hMemoryVirt,
            .virtualOffset  = lo,
            .size           = hi - lo,
         };
         util_dynarray_append_typed(&ctx->ops, struct nvkmd_nvrm_bind_op, op);
      }
   }
}

/* Mark [off, end) mapped, merging into va->bound (kept sorted, non-overlapping). */
static void
va_bound_add(struct nvkmd_nvrm_va *va, uint64_t off, uint64_t end)
{
   struct nvkmd_nvrm_va_range nr = { off, end };
   util_dynarray_append_typed(&va->bound, struct nvkmd_nvrm_va_range, nr);

   struct nvkmd_nvrm_va_range *arr = util_dynarray_begin(&va->bound);
   uint32_t n = util_dynarray_num_elements(&va->bound, struct nvkmd_nvrm_va_range);
   qsort(arr, n, sizeof(*arr), va_range_cmp);

   uint32_t w = 0;
   for (uint32_t i = 0; i < n; i++) {
      if (w > 0 && arr[i].start <= arr[w - 1].end) {
         if (arr[i].end > arr[w - 1].end)
            arr[w - 1].end = arr[i].end;
      } else {
         arr[w++] = arr[i];
      }
   }
   va->bound.size = w * sizeof(struct nvkmd_nvrm_va_range);
}

/* Mark [off, end) unmapped, splitting any straddling ranges in va->bound. */
static void
va_bound_remove(struct nvkmd_nvrm_va *va, uint64_t off, uint64_t end)
{
   struct util_dynarray out;
   util_dynarray_init(&out, NULL);

   util_dynarray_foreach(&va->bound, struct nvkmd_nvrm_va_range, r) {
      if (r->end <= off || r->start >= end) {
         util_dynarray_append_typed(&out, struct nvkmd_nvrm_va_range, *r);
         continue;
      }
      if (r->start < off) {
         struct nvkmd_nvrm_va_range left = { r->start, off };
         util_dynarray_append_typed(&out, struct nvkmd_nvrm_va_range, left);
      }
      if (r->end > end) {
         struct nvkmd_nvrm_va_range right = { end, r->end };
         util_dynarray_append_typed(&out, struct nvkmd_nvrm_va_range, right);
      }
   }

   util_dynarray_fini(&va->bound);
   va->bound = out;
}

static VkResult
nvkmd_nvrm_bind_ctx_bind(struct nvkmd_ctx *_ctx,
                            struct vk_object_base *log_obj,
                            uint32_t bind_count,
                            const struct nvkmd_ctx_bind *binds)
{
   struct nvkmd_nvrm_bind_ctx *ctx = nvkmd_nvrm_bind_ctx(_ctx);

   for (uint32_t i = 0; i < bind_count; i++) {
      const struct nvkmd_ctx_bind *b = &binds[i];
      struct nvkmd_nvrm_va *va = nvkmd_nvrm_va(b->va);
      uint64_t off = b->va_offset_B;
      uint64_t end = off + b->range_B;

      /* Clear whatever is already mapped in this range first; the mapper would
       * otherwise reject the MAP (rebind) or, for an unbind, we only touch the
       * pages that are actually mapped. */
      va_emit_unmap_overlaps(ctx, va, off, end);

      if (b->op == NVKMD_BIND_OP_BIND) {
         struct nvkmd_nvrm_mem *mem = nvkmd_nvrm_mem(b->mem);
         /* Match the synchronous bind path's GPU map flags: inherit the VA's
          * page kind, and snoop CPU caches for system memory so a CPU write to
          * the bound page is visible to the GPU (and vice versa). */
         uint32_t dmaFlags =
            DRF_DEF(OS46, _FLAGS, _PAGE_KIND, _VIRTUAL) |
            (mem->isSystemMem ? DRF_DEF(OS46, _FLAGS, _CACHE_SNOOP, _ENABLE)
                              : DRF_DEF(OS46, _FLAGS, _CACHE_SNOOP, _DISABLE));

         struct nvkmd_nvrm_bind_op op = {
            .unmap          = false,
            .hVirtualMemory = va->hMemoryVirt,
            .virtualOffset  = off,
            .hPhysicalMemory = mem->hMemoryPhys,
            .physicalOffset = b->mem_offset_B,
            .size           = b->range_B,
            .dmaFlags       = dmaFlags,
         };
         util_dynarray_append_typed(&ctx->ops, struct nvkmd_nvrm_bind_op, op);
         va_bound_add(va, off, end);
      } else {
         va_bound_remove(va, off, end);
      }
   }

   return VK_SUCCESS;
}

/* Queue one rendezvous chunk of paging ops on the mapper: SEMAPHORE_WAIT(GO),
 * the MAP/UNMAP ops, SEMAPHORE_SIGNAL(DONE), all keyed on `seq`. */
static VkResult
bind_submit_mapper_chunk(struct nvkmd_nvrm_bind_ctx *ctx,
                         struct vk_object_base *log_obj,
                         uint64_t seq,
                         const struct nvkmd_nvrm_bind_op *ops, uint32_t count)
{
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(ctx->base.dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   NV00FE_CTRL_SUBMIT_OPERATIONS_PARAMS *p = calloc(1, sizeof(*p));
   if (p == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   uint32_t k = 0;
   p->pOperations[k].type = NV00FE_CTRL_OPERATION_TYPE_SEMAPHORE_WAIT;
   p->pOperations[k].data.semaphore.index = NVRM_BIND_SLOT_GO;
   p->pOperations[k].data.semaphore.value = seq;
   k++;

   for (uint32_t i = 0; i < count; i++) {
      const struct nvkmd_nvrm_bind_op *op = &ops[i];
      NV00FE_CTRL_OPERATION *o = &p->pOperations[k++];
      if (op->unmap) {
         o->type = NV00FE_CTRL_OPERATION_TYPE_UNMAP;
         o->data.unmap.hVirtualMemory = op->hVirtualMemory;
         o->data.unmap.virtualOffset  = op->virtualOffset;
         o->data.unmap.size           = op->size;
         o->data.unmap.dmaFlags       = op->dmaFlags;
      } else {
         o->type = NV00FE_CTRL_OPERATION_TYPE_MAP;
         o->data.map.hVirtualMemory  = op->hVirtualMemory;
         o->data.map.virtualOffset   = op->virtualOffset;
         o->data.map.hPhysicalMemory = op->hPhysicalMemory;
         o->data.map.physicalOffset  = op->physicalOffset;
         o->data.map.size            = op->size;
         o->data.map.dmaFlags        = op->dmaFlags;
      }
   }

   p->pOperations[k].type = NV00FE_CTRL_OPERATION_TYPE_SEMAPHORE_SIGNAL;
   p->pOperations[k].data.semaphore.index = NVRM_BIND_SLOT_DONE;
   p->pOperations[k].data.semaphore.value = seq;
   k++;

   p->operationsCount = k;

   NV_STATUS nvRes = nvRmApiControl(&rm, ctx->hMemMapper,
                                    NV00FE_CTRL_CMD_SUBMIT_OPERATIONS,
                                    p, sizeof(*p));
   free(p);

   if (nvRes != NV_OK)
      return vk_errorf(log_obj, VK_ERROR_DEVICE_LOST,
                       "NV00FE_CTRL_CMD_SUBMIT_OPERATIONS failed: 0x%08x (%s)",
                       (unsigned)nvRes, nvkmd_nvrm_status_str(nvRes));

   return VK_SUCCESS;
}

static VkResult
nvkmd_nvrm_bind_ctx_flush(struct nvkmd_ctx *_ctx,
                             struct vk_object_base *log_obj)
{
   struct nvkmd_nvrm_bind_ctx *ctx = nvkmd_nvrm_bind_ctx(_ctx);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(ctx->base.dev);
   VkResult result = VK_SUCCESS;

   uint32_t n_ops = util_dynarray_num_elements(&ctx->ops, struct nvkmd_nvrm_bind_op);

   /* 1. Acquire the wait semaphores (the host stalls until the app signals). */
   util_dynarray_foreach(&ctx->waits, struct nvkmd_nvrm_bind_sem, w)
      write_semaphore_acquire(&ctx->chan.push, w->addr, w->value);
   chan_emit_push_segment(&ctx->chan);

   /* 2. Rendezvous the paging ops with the mapper, one mapper-queue-sized chunk
    *    at a time: release GO (+ interrupt) so the mapper runs, then acquire
    *    DONE so the channel stalls until the mapper finishes. */
   if (n_ops > 0) {
      uint64_t goAddr = bind_slot_gpu_addr(ctx, NVRM_BIND_SLOT_GO);
      uint64_t doneAddr = bind_slot_gpu_addr(ctx, NVRM_BIND_SLOT_DONE);
      /* A SUBMIT_OPERATIONS that fills the mapper queue to capacity is rejected
       * (NV_STATUS 0x55) -- it needs headroom -- and each chunk also spends a
       * WAIT + SIGNAL op, so cap a chunk comfortably below maxQueueSize. */
      uint32_t chunkMax = ctx->mapperQueueSize - 8;
      struct nvkmd_nvrm_bind_op *ops = util_dynarray_begin(&ctx->ops);

      for (uint32_t off = 0; off < n_ops; off += chunkMax) {
         uint32_t chunk = MIN2(chunkMax, n_ops - off);
         uint64_t seq = ++ctx->rendezvousSeq;

         write_semaphore_release(&ctx->chan.push, goAddr, seq, false);
         chan_emit_push_segment(&ctx->chan);

         result = bind_submit_mapper_chunk(ctx, log_obj, seq, &ops[off], chunk);
         if (result != VK_SUCCESS)
            goto done;

         write_semaphore_acquire(&ctx->chan.push, doneAddr, seq);
         chan_emit_push_segment(&ctx->chan);

         /* The mapper queue can't hold the next chunk's ops until this one has
          * drained, so when more remain, push this rendezvous to the GPU and
          * wait for the channel to consume the DONE acquire before queueing
          * more. Only large binds (> mapperQueueSize-2 ops) take this blocking
          * path; the common single-chunk case stays fully non-blocking. */
         if (off + chunkMax < n_ops) {
            result = nvrm_channel_submit(dev, &ctx->chan, log_obj);
            if (result != VK_SUCCESS)
               goto done;
            result = nvrm_channel_sync(&ctx->chan, log_obj);
            if (result != VK_SUCCESS)
               goto done;
         }
      }
   }

   /* 3. Release the signal semaphores (after DONE), waking their CPU waiters. */
   util_dynarray_foreach(&ctx->signals, struct nvkmd_nvrm_bind_sem, s)
      write_semaphore_release(&ctx->chan.push, s->addr, s->value, false);

   result = nvrm_channel_submit(dev, &ctx->chan, log_obj);

done:
   util_dynarray_clear(&ctx->waits);
   util_dynarray_clear(&ctx->ops);
   util_dynarray_clear(&ctx->signals);

   return result;
}

static VkResult
nvkmd_nvrm_bind_ctx_signal(struct nvkmd_ctx *_ctx,
                              struct vk_object_base *log_obj,
                              uint32_t signal_count,
                              const struct vk_sync_signal *signals)
{
   struct nvkmd_nvrm_bind_ctx *ctx = nvkmd_nvrm_bind_ctx(_ctx);

   for (uint32_t i = 0; i < signal_count; i++) {
      struct nvkmd_nvrm_bind_sem s = {
         .addr = nvkmd_nvrm_sync_gpu_addr(signals[i].sync),
         .value = nvkmd_nvrm_sync_gpu_signal_value(signals[i].sync, signals[i].signal_value),
      };
      util_dynarray_append_typed(&ctx->signals, struct nvkmd_nvrm_bind_sem, s);
   }

   return nvkmd_nvrm_bind_ctx_flush(&ctx->base, log_obj);
}

static VkResult
nvkmd_nvrm_bind_ctx_sync(struct nvkmd_ctx *_ctx,
                            struct vk_object_base *log_obj)
{
   struct nvkmd_nvrm_bind_ctx *ctx = nvkmd_nvrm_bind_ctx(_ctx);

   VkResult result = nvkmd_nvrm_bind_ctx_flush(&ctx->base, log_obj);
   if (result != VK_SUCCESS)
      return result;

   return nvrm_channel_sync(&ctx->chan, log_obj);
}

const struct nvkmd_ctx_ops nvkmd_nvrm_bind_ctx_ops = {
   .destroy = nvkmd_nvrm_bind_ctx_destroy,
   .wait = nvkmd_nvrm_bind_ctx_wait,
   .bind = nvkmd_nvrm_bind_ctx_bind,
   .signal = nvkmd_nvrm_bind_ctx_signal,
   .flush = nvkmd_nvrm_bind_ctx_flush,
   .sync = nvkmd_nvrm_bind_ctx_sync,
};

VkResult
nvkmd_nvrm_create_ctx(struct nvkmd_dev *dev,
                         struct vk_object_base *log_obj,
                         enum nvkmd_engines engines,
                         struct nvkmd_ctx **ctx_out)
{
   if (engines == NVKMD_ENGINE_BIND) {
      return nvkmd_nvrm_create_bind_ctx(dev, log_obj, ctx_out);
   } else {
      return nvkmd_nvrm_create_exec_ctx(dev, log_obj, engines, ctx_out);
   }
}
