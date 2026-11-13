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
#include "util/os_time.h"
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

/* Memory barrier for handing work to the GPU host.
 *
 * A submit writes three kinds of memory, mapped three different ways:
 *  - the pushbuffer and GPFIFO entries, in GART (sysmem, write-back cached,
 *    snooped by the GPU);
 *  - USERD GPPut, in vidmem through BAR1, which RM maps write-combined for
 *    user mappings of FB memory (mapping_cpu.c, RmMapMemory);
 *  - the doorbell, a register in the usermode page, mapped uncached.
 *
 * x86 keeps ordinary write-back stores in order, but write-combined stores
 * are weakly ordered: they sit in a WC buffer and may reach the device after
 * a later uncached store. Without a fence the doorbell can arrive before the
 * new GPPut, the host then reads the old GPPut, finds nothing new, and the
 * submission is left in the ring until some later doorbell. An application
 * that keeps submitting hides this (the next frame's doorbell picks the work
 * up); one that waits for the result of its last submission, or a channel
 * that submits rarely, such as the upload queue's copy channel, waits
 * forever.
 *
 * The compiler needs the barrier as much as the CPU: the GPFIFO entry and
 * pushbuffer stores are plain stores, which C lets it move past the volatile
 * GPPut and doorbell stores.
 *
 * NVIDIA's own submit path fences at both points: mb() between the GPFIFO
 * entry and GPPut (uvm_channel.c, "Need to make sure all the pushbuffer and
 * the GPFIFO entries writes complete before updating GPPUT"), and wmb()
 * between GPPut and the doorbell (uvm_volta_host.c,
 * uvm_hal_volta_host_write_gpu_put()). Mesa's amdgpu user-queue submit
 * (amdgpu_cs.cpp) uses mfence the same way before its write pointer and its
 * doorbell. mfence covers write-combined stores as well as ordinary ones,
 * which a locked instruction (what a C11 seq_cst fence may compile to) is
 * not documented to do for device memory, so use it explicitly. */
static inline void
chan_io_barrier(void)
{
#if defined(__x86_64__) || defined(__i386__)
	__asm__ volatile("mfence" ::: "memory");
#elif defined(__aarch64__)
	__asm__ volatile("dsb st" ::: "memory");
#else
	__sync_synchronize();
#endif
}

/* Tell the host the channel has new work. Re-reads GPPut from USERD, so
 * ringing it again with no new work is harmless. */
static void
chan_ring_doorbell(struct nvkmd_nvrm_channel *chan)
{
	struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(chan->dev->base.pdev);
	const volatile NvNotification *notifiers = chan->notifier->map;
	volatile NvU32 *doorbell = (volatile NvU32 *)
		((NvU8 *)pdev->usermodeMap.address + NVC361_NOTIFY_CHANNEL_PENDING);

	chan_io_barrier();
	*doorbell = notifiers[NV_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN].info32;
}

/* Completed-fence value: the highest wSeq the GPU has released into chan->sem. */
static uint64_t
chan_completed(struct nvkmd_nvrm_channel *chan)
{
	return *(volatile uint64_t *)chan->sem->map;
}

/* Report a channel wait that has run for seconds: the fence it needs, what
 * the GPU has completed, and where the GPFIFO stands, then ring the doorbell
 * again. GPGet stuck short of
 * GPPut with no RC error usually means the GPU is parked on a semaphore
 * acquire whose signal never came. */
static void
chan_report_stuck(struct nvkmd_nvrm_channel *chan, const char *what,
                  uint64_t target, uint64_t waited_ns)
{
	KeplerBControlGPFifo *userD = chan->userD->map;
	fprintf(stderr, "nvk/nvrm: %s waiting %.1f s for fence %" PRIu64
	        ": completed %" PRIu64 ", submitted %" PRIu64
	        ", GPGet %u GPPut %u\n",
	        what, waited_ns / 1e9, target, chan_completed(chan), chan->wSeq,
	        (unsigned)userD->GPGet, (unsigned)userD->GPPut);
	/* In case the stall is a lost doorbell (see chan_io_barrier()): ringing
	 * it again is harmless, and if the wait then ends, that was the cause. */
	chan_ring_doorbell(chan);
}

/* True once RM has reported a robust-channel error on this channel. */
static bool
chan_check_error(struct nvkmd_nvrm_channel *chan)
{
	volatile NvNotification *notifiers = chan->notifier->map;
	return notifiers[NV906F_NOTIFIERS_RC].status != 0;
}

/* RM's robust-channel error status for the channel: 0 while it is healthy.
 * A channel with an RC error has stopped executing for good. */
uint32_t
nvkmd_nvrm_channel_rc_status(const struct nvkmd_nvrm_channel *chan)
{
	const volatile NvNotification *notifiers = chan->notifier->map;
	return notifiers[NV906F_NOTIFIERS_RC].status;
}

/* Record a host semaphore method for nvkmd_nvrm_channel_report(). Methods
 * written now belong to the current submit cycle, which nvrm_channel_submit
 * retires with fence wSeq + 1. */
static void
chan_log_sem(struct nvkmd_nvrm_channel *chan, bool acquire,
             const volatile uint64_t *word, uint64_t addr, uint64_t value)
{
	struct nvkmd_nvrm_sem_record rec = {
		.word = word,
		.addr = addr,
		.value = value,
		.fence = chan->wSeq + 1,
	};
	simple_mtx_lock(&chan->dev->sync_mutex);
	if (acquire)
		chan->acqLog[chan->acqLogNext++ % NVRM_CHAN_SEM_LOG] = rec;
	else
		chan->relLog[chan->relLogNext++ % NVRM_CHAN_SEM_LOG] = rec;
	simple_mtx_unlock(&chan->dev->sync_mutex);
}

static void
chan_report_sem_log(const struct nvkmd_nvrm_channel *chan, bool acquire,
                    uint64_t completed)
{
	const struct nvkmd_nvrm_sem_record *recs =
		acquire ? chan->acqLog : chan->relLog;
	uint32_t next = acquire ? chan->acqLogNext : chan->relLogNext;
	/* Every acquire still pending (the oldest is the one the host is parked
	 * on), but only the latest releases. */
	uint32_t count = MIN2(next, acquire ? NVRM_CHAN_SEM_LOG : 6);

	for (uint32_t n = count; n > 0; n--) {
		const struct nvkmd_nvrm_sem_record *r =
			&recs[(next - n) % NVRM_CHAN_SEM_LOG];
		bool done = r->fence <= completed;
		if (acquire && done)
			continue;  /* only acquires the GPU may still be parked on */

		if (!done && r->word != NULL) {
			uint64_t now = *r->word;
			fprintf(stderr, "nvk/nvrm:     %s 0x%" PRIx64 " %s %" PRIu64
			        " (now %" PRIu64 "%s), fence %" PRIu64 ", pending\n",
			        acquire ? "acquire" : "release", r->addr,
			        acquire ? ">=" : "=", r->value, now,
			        acquire ? (now >= r->value ? ", satisfied"
			                                   : ", NOT satisfied") : "",
			        r->fence);
		} else {
			fprintf(stderr, "nvk/nvrm:     %s 0x%" PRIx64 " %s %" PRIu64
			        ", fence %" PRIu64 ", %s\n",
			        acquire ? "acquire" : "release", r->addr,
			        acquire ? ">=" : "=", r->value, r->fence,
			        done ? "done" : "pending");
		}
	}
}

/* Print the channel's state and its recent semaphore methods, for a CPU wait
 * that has stalled (nvkmd_nvrm_sync.c). Caller holds dev->sync_mutex. */
void
nvkmd_nvrm_channel_report(const struct nvkmd_nvrm_channel *chan)
{
	const KeplerBControlGPFifo *userD = chan->userD->map;
	uint64_t completed = nvkmd_nvrm_channel_completed(chan);
	uint32_t rc = nvkmd_nvrm_channel_rc_status(chan);

	/* USERD GPPut is what the host reads; "CPU" is what this process last
	 * wrote there. If they differ, the GPPut write has not reached the GPU. */
	fprintf(stderr, "nvk/nvrm:   %s channel %p: completed fence %" PRIu64
	        " of %" PRIu64 " submitted, GPGet %u GPPut %u (CPU %u)%s",
	        chan->kind ? chan->kind : "?", (const void *)chan, completed,
	        chan->wSeq, (unsigned)userD->GPGet, (unsigned)userD->GPPut,
	        (unsigned)(chan->gpPut % NVRM_CTX_GPFIFO_ENTRIES),
	        completed >= chan->wSeq ? ", idle" : "");
	if (rc != 0)
		fprintf(stderr, ", RC ERROR status 0x%x (channel stopped)", rc);
	fprintf(stderr, "\n");

	chan_report_sem_log(chan, true, completed);
	chan_report_sem_log(chan, false, completed);
}

/* Ring the doorbell of a channel that has submitted work the GPU has not
 * completed, for a CPU wait that has stalled. If the work then completes,
 * the host had not been told about it: a doorbell was lost. Returns whether
 * it rang. Caller holds dev->sync_mutex. */
bool
nvkmd_nvrm_channel_kick(struct nvkmd_nvrm_channel *chan)
{
	if (nvkmd_nvrm_channel_rc_status(chan) != 0)
		return false;
	if (nvkmd_nvrm_channel_completed(chan) >=
	    __atomic_load_n(&chan->wSeq, __ATOMIC_ACQUIRE))
		return false;
	chan_ring_doorbell(chan);
	return true;
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

/* The GPFIFO position the host is known to have consumed: the GPPut recorded
 * at the end of the newest submit cycle whose fence has completed (a completed
 * fence means every GP entry before it was fetched and executed).
 *
 * USERD GPGet is not used for this. On an RTX 5070 under Haiku the render
 * thread hung in the back-pressure loop below (waiting for GPGet to move)
 * a few seconds into rendering, while the slot-recycle wait, which reads the
 * channel fence, never stalled: GPGet as seen through the CPU mapping was not
 * keeping up with the GPU. The channel fence is the signal the slot-recycle
 * wait already relies on, so use it here too. */
static uint32_t
chan_retired_gp_get(struct nvkmd_nvrm_channel *chan)
{
	uint64_t done = chan_completed(chan);
	uint64_t best = 0;
	for (uint32_t i = 0; i < NVRM_CTX_CMDBUF_SLOTS; i++) {
		uint64_t f = chan->cmdBufSlotFence[i];
		if (f != 0 && f <= done && f > best) {
			best = f;
			chan->gpGet = chan->cmdBufSlotGpPut[i];
		}
	}
	/* chan->gpGet keeps the last known position when every recorded slot is
	 * still pending (retirement only moves forward). */
	return (uint32_t)(chan->gpGet % NVRM_CTX_GPFIFO_ENTRIES);
}

static void
chan_write_gp_fifo_entry(struct nvkmd_nvrm_channel *chan, const struct nvkmd_ctx_exec *exec)
{
	uint32_t next = (uint32_t)((chan->gpPut + 1) % NVRM_CTX_GPFIFO_ENTRIES);

	/* Back-pressure: never let GPPut catch the host's read position. Earlier
	 * submissions have been doorbelled, so the host keeps draining and the
	 * channel fence advances; bail out on a dead channel rather than spin
	 * forever. In practice the 32-slot cmdBuf ring throttles submission long
	 * before the 0x8000-entry GPFIFO fills, so this rarely waits at all. */
	const uint64_t wait_start = os_time_get_nano();
	uint64_t next_report = wait_start + 3000000000ull;
	while (next == chan_retired_gp_get(chan)) {
		if (chan_check_error(chan))
			break;
		uint64_t now = os_time_get_nano();
		if (now >= next_report) {
			chan_report_stuck(chan, "GPFIFO full", chan->wSeq, now - wait_start);
			next_report = now + 5000000000ull;
		}
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

uint64_t
nvkmd_nvrm_channel_completed(const struct nvkmd_nvrm_channel *chan)
{
	return *(volatile uint64_t *)chan->sem->map;
}

uint64_t
nvkmd_nvrm_channel_retire_point(const struct nvkmd_nvrm_channel *chan)
{
	/* Methods already in the push but not yet submitted will be retired by
	 * the next fence. Read the push before wSeq: nvrm_channel_submit()
	 * publishes the new wSeq before resetting the push, so a concurrent
	 * submit can make this one fence too high, never too low. */
	bool unsubmitted = nv_push_dw_count(&chan->push) > 0;
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	uint64_t submitted = __atomic_load_n(&chan->wSeq, __ATOMIC_ACQUIRE);
	return unsubmitted ? submitted + 1 : submitted;
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

   /* The channel is gone, so none of its work can touch a sync slot any more:
    * stop deferring slot reuse on it. Must come before chan->sem is released,
    * which the slot pool reads while the channel is registered. No-op for a
    * channel that never finished init (never registered). */
   nvkmd_nvrm_sync_unregister_channel(dev, chan);

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
 * Engine subchannel objects (if any) are the caller's responsibility.
 * engineType is the RM engine the channel runs on (NV2080_ENGINE_TYPE_*):
 * graphics for everything but video decode, which has its own engine. */
static VkResult
nvrm_channel_init(struct nvkmd_nvrm_dev *dev,
                  struct vk_object_base *log_obj,
                  struct nvkmd_nvrm_channel *chan,
                  NvU32 engineType)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct nvkmd_dev *_dev = &dev->base;
   VkResult vkRes;

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   chan->dev = dev;
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

   /* From here on its SEM methods may name sync-pool slots. An untracked
    * channel would let slots be reused under it, so fail instead. */
   if (!nvkmd_nvrm_sync_register_channel(dev, chan)) {
      vkRes = vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto error;
   }

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
   KeplerBControlGPFifo *userD = chan->userD->map;
   uint64_t semAdrGpu = chan->sem->va->addr;

   /* Published before the push is reset (chan_begin_slot below); see
    * nvkmd_nvrm_channel_retire_point(). */
   __atomic_store_n(&chan->wSeq, chan->wSeq + 1, __ATOMIC_RELEASE);

   /* Channel fence: the drained-state marker the slot-recycle wait polls on. */
   write_semaphore_release(&chan->push, semAdrGpu, chan->wSeq, true);
   chan_emit_push_segment(chan);

   /* The pushbuffer and GPFIFO entries must be in memory before the host can
    * see the new GPPut, and GPPut must be before the doorbell; see
    * chan_io_barrier(). chan_ring_doorbell() supplies the second barrier. */
   chan_io_barrier();
   userD->GPPut = chan->gpPut;
   chan_ring_doorbell(chan);

   chan->cmdBufSlotFence[chan->cmdBufCurSlot] = chan->wSeq;
   chan->cmdBufSlotGpPut[chan->cmdBufCurSlot] = (uint32_t)chan->gpPut;
   chan->cmdBufCurSlot = (chan->cmdBufCurSlot + 1) % NVRM_CTX_CMDBUF_SLOTS;

   VkResult result = VK_SUCCESS;
   const uint64_t wait_start = os_time_get_nano();
   uint64_t next_report = wait_start + 3000000000ull;
   while (chan_completed(chan) < chan->cmdBufSlotFence[chan->cmdBufCurSlot]) {
      if (chan_check_error(chan)) {
         result = vk_error(log_obj, VK_ERROR_DEVICE_LOST);
         break;
      }
      uint64_t now = os_time_get_nano();
      if (now >= next_report) {
         chan_report_stuck(chan, "cmdbuf slot recycle", chan->cmdBufSlotFence[chan->cmdBufCurSlot], now - wait_start);
         next_report = now + 5000000000ull;
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
   const uint64_t wait_start = os_time_get_nano();
   uint64_t next_report = wait_start + 3000000000ull;
   while (chan_completed(chan) < chan->wSeq) {
      if (chan_check_error(chan))
         return vk_error(log_obj, VK_ERROR_DEVICE_LOST);
      uint64_t now = os_time_get_nano();
      if (now >= next_report) {
         chan_report_stuck(chan, "channel sync", chan->wSeq, now - wait_start);
         next_report = now + 5000000000ull;
      }
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
   if (engines & NVKMD_ENGINE_VDEC) {
      /* Video decode (Mesa 26.3, VK_KHR_video_decode_queue): its own
       * channel on the NVDEC engine with the decoder object, as the nouveau
       * winsys makes it (nouveau_ws_context_create() with
       * NOUVEAU_WS_ENGINE_VDEC). NVK asks for it alone, for its video
       * decode queue family; an NVDEC channel can't also carry graphics,
       * so refuse a mix. The host semaphore methods the rest of this file
       * emits are host (channel) methods and work on any engine's channel.
       * NVDEC0 is the first decoder; GPUs with more decode on the first. */
      if (engines & ~NVKMD_ENGINE_VDEC) {
         vkRes = vk_errorf(log_obj, VK_ERROR_FEATURE_NOT_PRESENT,
                           "nvrm: video decode can't share a channel "
                           "with engines %#x", engines & ~NVKMD_ENGINE_VDEC);
         goto error;
      }
      if (pdev->base.dev_info.cls_vdec == 0) {
         vkRes = vk_errorf(log_obj, VK_ERROR_FEATURE_NOT_PRESENT,
                           "nvrm: no video decoder class on this GPU");
         goto error;
      }
      ctx->chan.kind = "vdec";
      VK_CHECK(nvrm_channel_init(dev, log_obj, &ctx->chan,
                                 NV2080_ENGINE_TYPE_NVDEC0));
      NV_CHECK(nvRmApiAlloc(&rm, ctx->chan.hChannel, &ctx->subchannels.hVdec,
                            pdev->base.dev_info.cls_vdec, NULL));
      *ctx_out = &ctx->base;
      return VK_SUCCESS;
   }

   ctx->chan.kind = engines == NVKMD_ENGINE_COPY ? "copy" : "exec";

   VK_CHECK(nvrm_channel_init(dev, log_obj, &ctx->chan,
                              NV2080_ENGINE_TYPE_GRAPHICS));

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
   if (ctx->subchannels.hVdec != 0)
      nvRmApiFree(&rm, ctx->subchannels.hVdec);

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
      chan_log_sem(&ctx->chan, true, nvkmd_nvrm_sync_cpu_word(waits[i].sync),
                   addr, value);
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
      chan_log_sem(&ctx->chan, false, nvkmd_nvrm_sync_cpu_word(signals[i].sync),
                   addr, value);
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
   ctx->chan.kind = "bind";
   ctx->mapperQueueSize = NVRM_BIND_MAPPER_QUEUE_SIZE;
   util_dynarray_init(&ctx->waits, NULL);
   util_dynarray_init(&ctx->signals, NULL);
   util_dynarray_init(&ctx->ops, NULL);

   VK_CHECK(nvrm_channel_init(dev, log_obj, &ctx->chan,
                              NV2080_ENGINE_TYPE_GRAPHICS));

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
         .word = nvkmd_nvrm_sync_cpu_word(waits[i].sync),
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
   util_dynarray_foreach(&ctx->waits, struct nvkmd_nvrm_bind_sem, w) {
      write_semaphore_acquire(&ctx->chan.push, w->addr, w->value);
      chan_log_sem(&ctx->chan, true, w->word, w->addr, w->value);
   }
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
         chan_log_sem(&ctx->chan, true, NULL, doneAddr, seq);
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
   util_dynarray_foreach(&ctx->signals, struct nvkmd_nvrm_bind_sem, s) {
      write_semaphore_release(&ctx->chan.push, s->addr, s->value, false);
      chan_log_sem(&ctx->chan, false, s->word, s->addr, s->value);
   }

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
         .word = nvkmd_nvrm_sync_cpu_word(signals[i].sync),
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
