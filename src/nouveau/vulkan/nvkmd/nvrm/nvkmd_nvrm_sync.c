/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nvrm.h"

#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <stdlib.h>

#include "vk_log.h"
#include "nvk_device.h"

#include "util/u_memory.h"
#include "util/macros.h"
#include "util/os_time.h"

#include "class/cl0005.h"               // NV01_EVENT_OS_EVENT, NV01_EVENT_NONSTALL_INTR
#include "class/cl2080_notification.h"  // NV2080_NOTIFIERS_FIFO_EVENT_MTHD
#include "ctrl/ctrl2080/ctrl2080event.h" // NV2080_CTRL_CMD_EVENT_SET_TRIGGER_FIFO


#define NVRM_SYNC_DEBUG 0

/* Poll interval for the polling fallback (no wake event available, e.g.
 * WAIT_PENDING). */
#define NVRM_SYNC_POLL_US 250

/* Per-sync backing state.
 *
 * `value` is the semaphore payload the GPU releases to / acquires on and the
 * CPU reads. The remaining words are a coordination header. They live
 * alongside the payload in the pool slot or, for shareable syncs, in the
 * dedicated page that gets exported -- so the header is shared with importers.
 * OPAQUE_FD guarantees an importer is this same driver, so the layout is a
 * private ABI we fully control.
 *
 * `submit_seq` tracks the highest *submitted* payload value, advanced when a
 * signal is emitted (before the GPU release that completes it writes `value`).
 * It is what VK_SYNC_WAIT_PENDING resolves against; `value` is what a normal
 * (completion) wait resolves against.
 *
 * Binary scheme: the payload is monotonic. Each signal writes a fresh
 * `submit_seq` (++ per signal); each wait acquires its own `wait_seq`. Since
 * binary semaphore signal/wait pairs are 1:1 and ordered, the Nth wait pairs
 * the Nth signal and acquiring `>= wait_seq` matches exactly the value that
 * signal writes -- even for wait-before-signal. `reset` then never lowers
 * `value` (which would strand an in-flight GPU acquire); it raises `reset_mark`
 * so the CPU-visible state (signaled iff value > reset_mark) reads unsignaled
 * until a fresh signal. */
struct nvrm_sync_data {
   uint64_t value;       /* completed payload: GPU release / CPU signal writes it */
   uint64_t submit_seq;  /* highest submitted payload value (WAIT_PENDING target) */
   uint64_t wait_seq;    /* binary: ++ per emitted wait; the value it acquires */
   uint64_t reset_mark;  /* binary CPU state: signaled iff value > reset_mark */
};

/* One slot per sync, sized to hold nvrm_sync_data and keep slots naturally
 * aligned within a page (page base is page-aligned). */
#define NVRM_SYNC_SLOT_SIZE sizeof(struct nvrm_sync_data)


/*
 * Semaphore-word slot pool (non-shareable syncs)
 *
 * Each page is one GART allocation holding num_slots words. A used-slot bitmap
 * tracks occupancy; pages grow lazily and are freed only at device teardown.
 */

struct nvkmd_nvrm_sync_page {
   struct nvkmd_mem *mem;
   uint32_t num_slots;
   uint32_t free_count;
   uint64_t bitmap[];  /* used-slot bits; DIV_ROUND_UP(num_slots, 64) words */
};

/* An OS wake event (its fd plus the NV01_EVENT bound to it) pooled for reuse
 * by CPU waits. */
struct nvkmd_nvrm_sync_event {
   int fd;
   NvHandle hEvent;
};

static void sync_event_destroy(struct nvkmd_nvrm_dev *dev, int fd,
                               NvHandle hEvent);

static struct nvkmd_nvrm_sync *
to_nvkmd_nvrm_sync(struct vk_sync *sync)
{
   assert(vk_sync_type_is_nvkmd_nvrm_sync(sync->type));
   return container_of(sync, struct nvkmd_nvrm_sync, base);
}

static struct nvrm_sync_data *
sync_data(struct nvkmd_nvrm_sync *sync)
{
   return (struct nvrm_sync_data *)((uint8_t *)sync->mem->map + sync->offset);
}

static bool
sync_is_timeline(const struct vk_sync *sync)
{
   return sync->flags & VK_SYNC_IS_TIMELINE;
}

/* Notify all non-stall (HOST engine) events for the subdevice so blocked CPU
 * waiters re-check their values. A CPU write to a semaphore word raises no GPU
 * interrupt, so this is how a CPU signal reaches waiters; GPU releases reach
 * them via their own NON_STALL_INTERRUPT. */
static void
sync_trigger_fifo(struct nvkmd_nvrm_dev *dev)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   NV2080_CTRL_EVENT_SET_TRIGGER_FIFO_PARAMS params = { .hEvent = 0 };
   nvRmApiControl(&rm, pdev->hSubdevice,
                  NV2080_CTRL_CMD_EVENT_SET_TRIGGER_FIFO,
                  &params, sizeof(params));
}

static uint32_t
sync_pool_slots_per_page(void)
{
   return 0x1000 / NVRM_SYNC_SLOT_SIZE;
}

static VkResult
sync_pool_alloc_slot(struct nvkmd_nvrm_dev *dev,
                     struct nvkmd_mem **mem_out, uint64_t *offset_out)
{
   VkResult result = VK_SUCCESS;
   simple_mtx_lock(&dev->sync_mutex);

   uint32_t per_page = sync_pool_slots_per_page();

   struct nvkmd_nvrm_sync_page *page = NULL;
   util_dynarray_foreach(&dev->sync_pages, struct nvkmd_nvrm_sync_page *, it) {
      if ((*it)->free_count > 0) {
         page = *it;
         break;
      }
   }

   if (page == NULL) {
      uint32_t words = DIV_ROUND_UP(per_page, 64);
      page = calloc(1, sizeof(*page) + words * sizeof(uint64_t));
      if (page == NULL) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto out;
      }
      VkResult vkRes = nvkmd_dev_alloc_mapped_mem(&dev->base, NULL, 0x1000, 0x1000,
                                                  NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR,
                                                  &page->mem);
      if (vkRes != VK_SUCCESS) {
         free(page);
         result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
         goto out;
      }
      memset(page->mem->map, 0, page->mem->size_B);
      page->num_slots = per_page;
      page->free_count = per_page;
      util_dynarray_append_typed(&dev->sync_pages,
                                 struct nvkmd_nvrm_sync_page *, page);
   }

   uint32_t slot = 0;
   while (page->bitmap[slot / 64] & (1ull << (slot % 64)))
      slot++;
   assert(slot < page->num_slots);

   page->bitmap[slot / 64] |= (1ull << (slot % 64));
   page->free_count--;

   /* Clear any stale state left by a previous occupant (payload and header). */
   memset((uint8_t *)page->mem->map + (uint64_t)slot * NVRM_SYNC_SLOT_SIZE,
          0, NVRM_SYNC_SLOT_SIZE);

   *mem_out = page->mem;
   *offset_out = (uint64_t)slot * NVRM_SYNC_SLOT_SIZE;

out:
   simple_mtx_unlock(&dev->sync_mutex);
   return result;
}

static void
sync_pool_free_slot(struct nvkmd_nvrm_dev *dev,
                    struct nvkmd_mem *mem, uint64_t offset)
{
   uint32_t slot = (uint32_t)(offset / NVRM_SYNC_SLOT_SIZE);

   simple_mtx_lock(&dev->sync_mutex);
   util_dynarray_foreach(&dev->sync_pages, struct nvkmd_nvrm_sync_page *, it) {
      struct nvkmd_nvrm_sync_page *page = *it;
      if (page->mem == mem) {
         assert(page->bitmap[slot / 64] & (1ull << (slot % 64)));
         page->bitmap[slot / 64] &= ~(1ull << (slot % 64));
         page->free_count++;
         break;
      }
   }
   simple_mtx_unlock(&dev->sync_mutex);
}

void
nvkmd_nvrm_sync_pool_finish(struct nvkmd_nvrm_dev *dev)
{
   util_dynarray_foreach(&dev->sync_events, struct nvkmd_nvrm_sync_event, ev)
      sync_event_destroy(dev, ev->fd, ev->hEvent);
   util_dynarray_fini(&dev->sync_events);

   util_dynarray_foreach(&dev->sync_pages, struct nvkmd_nvrm_sync_page *, it) {
      struct nvkmd_nvrm_sync_page *page = *it;
      nvkmd_mem_unref(page->mem);
      free(page);
   }
   util_dynarray_fini(&dev->sync_pages);
}

/* Release a sync's backing (pool slot back to the bitmap, or a dedicated mem
 * unreffed). Leaves sync->mem NULL. */
static void
sync_release_backing(struct nvkmd_nvrm_sync *sync)
{
   if (sync->mem == NULL)
      return;
   if (sync->dedicated)
      nvkmd_mem_unref(sync->mem);
   else
      sync_pool_free_slot(sync->dev, sync->mem, sync->offset);
   sync->mem = NULL;
}


/*
 * vk_sync implementation
 *
 * Each sync is backed by an nvrm_sync_data (see top of file). For a timeline
 * `value` is the user value directly; for a binary sync `value` is a monotonic
 * payload and the signaled/unsignaled state is derived from the header.
 */

static VkResult
nvkmd_nvrm_sync_init(struct vk_device *_device,
                    struct vk_sync *_sync,
                    uint64_t initial_value)
{
   struct nvk_device *vkDev = container_of(_device, struct nvk_device, vk);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(vkDev->nvkmd);
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);

   sync->dev = dev;
   sync->mem = NULL;
   sync->offset = 0;
   sync->dedicated = false;

   if (_sync->flags & VK_SYNC_IS_SHAREABLE) {
      /* Shareable syncs need their own mem so the backing word can be exported
       * as an opaque fd without exposing neighbouring slots. */
      VkResult vkRes = nvkmd_dev_alloc_mapped_mem(&dev->base, NULL, 0x1000, 0x1000,
                                                  NVKMD_MEM_GART | NVKMD_MEM_SHARED,
                                                  NVKMD_MEM_MAP_RDWR, &sync->mem);
      if (vkRes != VK_SUCCESS)
         return vk_error(_device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      memset(sync->mem->map, 0, sync->mem->size_B);
      sync->dedicated = true;
   } else {
      VkResult result = sync_pool_alloc_slot(dev, &sync->mem, &sync->offset);
      if (result != VK_SUCCESS)
         return vk_error(_device, result);
   }

   /* The backing memory is freshly zeroed (pool slot cleared on alloc, or the
    * dedicated page memset above), so submit_seq/wait_seq/reset_mark start at
    * 0. Set the payload: a timeline's value directly (submitted == completed at
    * init), or -- for a binary sync created already-signaled (e.g. a fence with
    * VK_FENCE_CREATE_SIGNALED_BIT, initial_value != 0) -- one completed signal
    * so value (1) > reset_mark (0) reads as signaled. */
   struct nvrm_sync_data *d = sync_data(sync);
   if (sync_is_timeline(_sync)) {
      d->value = initial_value;
      d->submit_seq = initial_value;
   } else if (initial_value != 0) {
      d->submit_seq = 1;
      d->value = 1;
   }

   return VK_SUCCESS;
}

void
nvkmd_nvrm_sync_finish(struct vk_device *_device,
                      struct vk_sync *_sync)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);

   sync_release_backing(sync);
}

/* Raise submit_seq (the highest submitted payload value) to at least `v`.
 * Timeline values are non-decreasing per the spec but may be emitted from
 * several threads, so use an atomic max rather than a plain store. */
static void
sync_record_submit(struct nvrm_sync_data *d, uint64_t v)
{
   uint64_t cur = __atomic_load_n(&d->submit_seq, __ATOMIC_ACQUIRE);
   while (v > cur &&
          !__atomic_compare_exchange_n(&d->submit_seq, &cur, v, true,
                                       __ATOMIC_SEQ_CST, __ATOMIC_ACQUIRE))
      ;
}

/* Apply a CPU-side signal to the payload. A timeline takes the value directly;
 * a binary sync advances the monotonic payload by handing out a fresh signal
 * sequence number, matching how a GPU release signals it (see
 * nvkmd_nvrm_sync_gpu_signal_value) so the payload never decreases. A CPU
 * signal completes immediately, so the submitted and completed values coincide. */
static void
sync_cpu_signal_data(struct nvkmd_nvrm_sync *sync, uint64_t value)
{
   struct nvrm_sync_data *d = sync_data(sync);
   uint64_t v;
   if (sync_is_timeline(&sync->base)) {
      v = value;
      sync_record_submit(d, v);
   } else {
      v = __atomic_add_fetch(&d->submit_seq, 1, __ATOMIC_SEQ_CST);
   }
   __atomic_store_n(&d->value, v, __ATOMIC_SEQ_CST);
}

/* Reset a binary sync to unsignaled without lowering the payload. Zeroing
 * `value` would race a GPU acquire emitted by an earlier submit that re-reads
 * the slot at execution time (the runtime resets a binary wait-semaphore right
 * after vkQueueSubmit in IMMEDIATE mode), stalling the host on that acquire
 * forever. Instead raise the baseline to the current payload, so "signaled"
 * (value > reset_mark) requires a fresh signal while the monotonic value stays
 * put for any in-flight acquire. */
static void
sync_reset_data(struct nvkmd_nvrm_sync *sync)
{
   struct nvrm_sync_data *d = sync_data(sync);
   uint64_t v = __atomic_load_n(&d->value, __ATOMIC_ACQUIRE);
   __atomic_store_n(&d->reset_mark, v, __ATOMIC_SEQ_CST);
}

static VkResult
nvkmd_nvrm_sync_signal(struct vk_device *_device,
                      struct vk_sync *_sync,
                      uint64_t value)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);

   sync_cpu_signal_data(sync, value);
   sync_trigger_fifo(sync->dev);

   return VK_SUCCESS;
}

static VkResult
nvkmd_nvrm_sync_get_value(struct vk_device *_device,
                         struct vk_sync *_sync,
                         uint64_t *value)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);
   struct nvrm_sync_data *d = sync_data(sync);

   uint64_t v = __atomic_load_n(&d->value, __ATOMIC_ACQUIRE);
   if (sync_is_timeline(_sync)) {
      *value = v;
   } else {
      uint64_t mark = __atomic_load_n(&d->reset_mark, __ATOMIC_ACQUIRE);
      *value = (v > mark) ? 1 : 0;
   }

   return VK_SUCCESS;
}

static VkResult
nvkmd_nvrm_sync_reset(struct vk_device *_device,
                     struct vk_sync *_sync)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);

   assert(!sync_is_timeline(_sync));  /* reset is a non-timeline operation */
   sync_reset_data(sync);

   return VK_SUCCESS;
}

static bool
sync_wait_satisfied(const struct vk_sync_wait *waits, uint32_t wait_count,
                    enum vk_sync_wait_flags wait_flags)
{
   bool wait_any = wait_flags & VK_SYNC_WAIT_ANY;
   bool pending = wait_flags & VK_SYNC_WAIT_PENDING;
   bool all = true;

   for (uint32_t i = 0; i < wait_count; i++) {
      struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(waits[i].sync);

      /* The sync word is GPU-written into GART that is not mapped coherent, so
       * invalidate the CPU cache for the page before reading. Without this the
       * poll can spin forever on a stale cached value even though the GPU has
       * advanced it. No-op for COHERENT memory. */
      nvkmd_mem_sync_map_from_gpu(sync->mem, 0, sync->mem->size_B);

      struct nvrm_sync_data *d = sync_data(sync);

      /* The default resolves against the completed payload (value), advanced
       * when the GPU executes the release. WAIT_PENDING additionally accepts a
       * merely-submitted signal, so the runtime's submit-time wait
       * (vk_queue_submit_move_binary_waits_to_temps) doesn't deadlock waiting
       * for completion of work gated behind that same submit.
       *
       * Submission is tracked in submit_seq, advanced (CPU-side) when a signal
       * is emitted. But submit_seq is written through the signaller's mapping;
       * across a *separate* mapping of imported memory that CPU write may not be
       * visible, whereas the GPU-written value reaches every mapping. So take
       * the max: completion implies submission, and value covers the imported
       * case that submit_seq cannot. */
      uint64_t value = __atomic_load_n(&d->value, __ATOMIC_ACQUIRE);
      uint64_t cur = value;
      if (pending) {
         uint64_t submitted = __atomic_load_n(&d->submit_seq, __ATOMIC_ACQUIRE);
         cur = MAX2(submitted, value);
      }

      bool reached;
      if (sync_is_timeline(waits[i].sync)) {
         reached = cur >= waits[i].wait_value;
      } else {
         /* Binary: signaled once the payload (or submitted count) has advanced
          * past the last reset baseline. */
         uint64_t mark = __atomic_load_n(&d->reset_mark, __ATOMIC_ACQUIRE);
         reached = cur > mark;
      }

      if (reached) {
         if (wait_any)
            return true;
      } else {
         all = false;
      }
   }

   return !wait_any && all;
}

/* Create a bare OS-event fd bound to the subdevice HOST non-stall interrupt,
 * used as the wake source for CPU waits. RM signals it on every HOST non-stall
 * interrupt (GPU semaphore release) and on every NV2080_CTRL_CMD_EVENT_SET_
 * TRIGGER_FIFO (CPU signal); the waiter re-checks its values on each wake. The
 * NV_ESC_ALLOC_OS_EVENT ioctl must run on the same fd being registered, so the
 * fd and its event are kept together and reused as a unit (see
 * sync_event_acquire). */
static int
sync_event_create(struct nvkmd_nvrm_dev *dev, NvHandle *hEvent_out)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   int fd = open(rm.nodeName, O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return -1;

   struct NvRmApi rmEv = rm;
   rmEv.fd = fd;
   if (nvRmApiAllocOsEvent(&rmEv, fd) != NV_OK) {
      close(fd);
      return -1;
   }

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
      .data = (NvP64)(uintptr_t)fd,
   };
   NvHandle hEvent = 0;
   if (nvRmApiAlloc(&rm, pdev->hSubdevice, &hEvent, NV01_EVENT_OS_EVENT, &eventParams) != NV_OK) {
      nvRmApiFreeOsEvent(&rmEv, fd);
      close(fd);
      return -1;
   }

   *hEvent_out = hEvent;
   return fd;
}

static void
sync_event_destroy(struct nvkmd_nvrm_dev *dev, int fd, NvHandle hEvent)
{
   if (fd < 0)
      return;

   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   nvRmApiFree(&rm, hEvent);

   struct NvRmApi rmEv = rm;
   rmEv.fd = fd;
   nvRmApiFreeOsEvent(&rmEv, fd);
   close(fd);
}

/* Obtain a wake event for a CPU wait: reuse an idle one from the pool, or
 * create a fresh one if the pool is empty (the create path runs the RM ioctls
 * outside the lock). A reused event may carry a stale one-shot wake latch, so
 * the first poll on it can return immediately; the caller re-checks its values
 * on every wake, so that is harmless. Returns the fd, or -1 on failure. */
static int
sync_event_acquire(struct nvkmd_nvrm_dev *dev, NvHandle *hEvent_out)
{
   simple_mtx_lock(&dev->sync_mutex);
   if (dev->sync_events.size != 0) {
      struct nvkmd_nvrm_sync_event ev =
         util_dynarray_pop(&dev->sync_events, struct nvkmd_nvrm_sync_event);
      simple_mtx_unlock(&dev->sync_mutex);
      *hEvent_out = ev.hEvent;
      return ev.fd;
   }
   simple_mtx_unlock(&dev->sync_mutex);

   return sync_event_create(dev, hEvent_out);
}

/* Return a wake event to the pool for reuse. */
static void
sync_event_release(struct nvkmd_nvrm_dev *dev, int fd, NvHandle hEvent)
{
   if (fd < 0)
      return;

   struct nvkmd_nvrm_sync_event ev = { .fd = fd, .hEvent = hEvent };
   simple_mtx_lock(&dev->sync_mutex);
   util_dynarray_append_typed(&dev->sync_events,
                              struct nvkmd_nvrm_sync_event, ev);
   simple_mtx_unlock(&dev->sync_mutex);
}

static VkResult
sync_wait_impl(struct nvkmd_nvrm_dev *dev,
               uint32_t wait_count,
               const struct vk_sync_wait *waits,
               enum vk_sync_wait_flags wait_flags,
               uint64_t abs_timeout_ns)
{
   if (wait_count == 0)
      return VK_SUCCESS;
   if (sync_wait_satisfied(waits, wait_count, wait_flags))
      return VK_SUCCESS;
   if (abs_timeout_ns == 0)
      return VK_TIMEOUT;

   /* WAIT_PENDING resolves against submitted values rather than the completed
    * value the word carries, so there is no wake event for it: poll. */
   bool can_event = !(wait_flags & VK_SYNC_WAIT_PENDING);
   NvHandle hEvent = 0;
   int evFd = can_event ? sync_event_acquire(dev, &hEvent) : -1;

   VkResult result;
   for (;;) {
      if (sync_wait_satisfied(waits, wait_count, wait_flags)) {
         result = VK_SUCCESS;
         break;
      }

      int64_t now = os_time_get_nano();
      if ((uint64_t)now >= abs_timeout_ns) {
         result = VK_TIMEOUT;
         break;
      }

      /* Unsigned, so a near-UINT64_MAX (effectively infinite) timeout doesn't
       * overflow into a negative span and collapse the wait into a tight spin;
       * each path caps the span to its own re-check interval anyway. */
      uint64_t remaining_ns = abs_timeout_ns - (uint64_t)now;
      if (evFd >= 0) {
         /* Cap the block so a missed wakeup still re-checks the values. */
         uint64_t budget_ms = remaining_ns / 1000000;
         if (budget_ms > 100)
            budget_ms = 100;
         struct pollfd pfd = { .fd = evFd, .events = POLLIN | POLLPRI };
         poll(&pfd, 1, budget_ms > 0 ? (int)budget_ms : 1);
      } else {
         uint64_t sleep_us = remaining_ns / 1000;
         if (sleep_us > NVRM_SYNC_POLL_US)
            sleep_us = NVRM_SYNC_POLL_US;
         if (sleep_us < 1)
            sleep_us = 1;
         os_time_sleep(sleep_us);
      }
   }

   sync_event_release(dev, evFd, hEvent);

   return result;
}

static VkResult
nvkmd_nvrm_sync_wait_many(struct vk_device *_device,
                         uint32_t wait_count,
                         const struct vk_sync_wait *waits,
                         enum vk_sync_wait_flags wait_flags,
                         uint64_t abs_timeout_ns)
{
   struct nvk_device *vkDev = container_of(_device, struct nvk_device, vk);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(vkDev->nvkmd);

   return sync_wait_impl(dev, wait_count, waits, wait_flags, abs_timeout_ns);
}

static VkResult
nvkmd_nvrm_sync_import_opaque_fd(struct vk_device *device,
                                struct vk_sync *_sync,
                                int fd)
{
   struct nvk_device *vkDev = container_of(device, struct nvk_device, vk);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(vkDev->nvkmd);
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);

   /* The opaque fd carries the semaphore-word backing memory. Import it and
    * map it for CPU access. */
   struct nvkmd_mem *mem = NULL;
   VkResult vkRes = nvkmd_dev_import_dma_buf(&dev->base, NULL, fd, &mem);
   if (vkRes != VK_SUCCESS)
      return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   if (mem->map == NULL) {
      void *mapOut = NULL;
      vkRes = nvkmd_mem_map(mem, NULL, NVKMD_MEM_MAP_RDWR, NULL, &mapOut);
      if (vkRes != VK_SUCCESS) {
         nvkmd_mem_unref(mem);
         return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      }
   }

   /* Swap in the imported mem, dropping the old backing. */
   sync_release_backing(sync);
   sync->mem = mem;
   sync->offset = 0;
   sync->dedicated = true;

   return VK_SUCCESS;
}

static VkResult
nvkmd_nvrm_sync_export_opaque_fd(struct vk_device *device,
                                struct vk_sync *_sync,
                                int *fd)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);

   /* Only dedicated (shareable) syncs own their backing memory; a pooled slot
    * shares a page with other syncs and can't be handed out. */
   if (sync->mem == NULL || !sync->dedicated)
      return vk_errorf(device, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "sync is not exportable");

   return nvkmd_mem_export_dma_buf(sync->mem, NULL, fd);
}

static VkResult
nvkmd_nvrm_sync_move(struct vk_device *_device,
                    struct vk_sync *_dst,
                    struct vk_sync *_src)
{
   struct nvkmd_nvrm_sync *dst = to_nvkmd_nvrm_sync(_dst);
   struct nvkmd_nvrm_sync *src = to_nvkmd_nvrm_sync(_src);

   if (src->dedicated) {
      /* A dedicated (shareable/imported) sync's identity is its backing memory,
       * which it may share with an external semaphore -- so it must not be
       * relocated. Hand dst a *reference* to the same backing (so the moved
       * wait's GPU acquire reads the live payload) and leave src bound to it,
       * merely reset. Reset only raises reset_mark and never lowers value, so
       * dst's acquire still sees the signalled value, and because src keeps the
       * shared memory a later signal of the external semaphore is still
       * observed through src. */
      sync_release_backing(dst);
      dst->mem = nvkmd_mem_ref(src->mem);
      dst->offset = src->offset;
      dst->dedicated = true;

      sync_reset_data(src);
      return VK_SUCCESS;
   }

   /* Pooled src: its slot is fungible, so swap backings. dst takes src's slot
    * (with any in-flight GPU signal targeting it) and src takes dst's old slot,
    * reset to unsignaled. Keeps both syncs valid without allocating. */
   struct nvkmd_mem *tmpMem = dst->mem;
   uint64_t tmpOffset = dst->offset;
   bool tmpDedicated = dst->dedicated;

   dst->mem = src->mem;
   dst->offset = src->offset;
   dst->dedicated = src->dedicated;

   src->mem = tmpMem;
   src->offset = tmpOffset;
   src->dedicated = tmpDedicated;

   if (src->mem != NULL)
      sync_reset_data(src);

   return VK_SUCCESS;
}


void
nvkmd_nvrm_sync_cpu_signal(struct vk_sync *_sync, uint64_t value)
{
   struct nvkmd_nvrm_sync *sync = vk_sync_as_nvkmd_nvrm_sync(_sync);
   if (sync == NULL || sync->mem == NULL)
      return;

   sync_cpu_signal_data(sync, value);
   sync_trigger_fifo(sync->dev);
}

uint64_t
nvkmd_nvrm_sync_gpu_addr(struct vk_sync *_sync)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);
   return sync->mem->va->addr + sync->offset;
}

/* Value a GPU release writes to signal this sync. For a binary sync this is a
 * fresh sequence number: the payload only ever increases, so a later reset
 * (which raises reset_mark, never lowers value) cannot retroactively break an
 * acquire that paired with this signal. Either way submit_seq is advanced now,
 * at emit time, so WAIT_PENDING sees the signal as submitted before the GPU
 * release executes and writes `value`. */
uint64_t
nvkmd_nvrm_sync_gpu_signal_value(struct vk_sync *_sync, uint64_t value)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);
   struct nvrm_sync_data *d = sync_data(sync);
   if (sync_is_timeline(_sync)) {
      sync_record_submit(d, value);
      return value;
   }
   return __atomic_add_fetch(&d->submit_seq, 1, __ATOMIC_SEQ_CST);
}

/* Value a GPU acquire waits for to consume a signal of this sync. Binary
 * semaphore signal/wait operations are 1:1 and ordered, so the Nth wait pairs
 * the Nth signal; acquiring >= the wait's own sequence number therefore matches
 * exactly the value that signal writes, regardless of which is emitted first
 * (wait-before-signal just stalls the host until the signal lands). The
 * counters live in the (possibly shared) backing memory so an imported sync's
 * two ends stay in lockstep. */
uint64_t
nvkmd_nvrm_sync_gpu_wait_value(struct vk_sync *_sync, uint64_t value)
{
   struct nvkmd_nvrm_sync *sync = to_nvkmd_nvrm_sync(_sync);
   if (sync_is_timeline(_sync))
      return value;
   return __atomic_add_fetch(&sync_data(sync)->wait_seq, 1, __ATOMIC_SEQ_CST);
}


struct vk_sync_type
nvkmd_nvrm_sync_get_type(struct nvkmd_nvrm_pdev *pdev)
{
   struct vk_sync_type type = {
      .size = sizeof(struct nvkmd_nvrm_sync),
      .features = VK_SYNC_FEATURE_BINARY |
                  VK_SYNC_FEATURE_GPU_WAIT |
                  VK_SYNC_FEATURE_CPU_RESET |
                  VK_SYNC_FEATURE_CPU_SIGNAL |
                  VK_SYNC_FEATURE_WAIT_PENDING |
                  VK_SYNC_FEATURE_CPU_WAIT |
                  VK_SYNC_FEATURE_TIMELINE,
      .init = nvkmd_nvrm_sync_init,
      .finish = nvkmd_nvrm_sync_finish,
      .signal = nvkmd_nvrm_sync_signal,
      .get_value = nvkmd_nvrm_sync_get_value,
      .reset = nvkmd_nvrm_sync_reset,
      .move = nvkmd_nvrm_sync_move,
      .wait_many = nvkmd_nvrm_sync_wait_many,
      .import_opaque_fd = nvkmd_nvrm_sync_import_opaque_fd,
      .export_opaque_fd = nvkmd_nvrm_sync_export_opaque_fd,
   };
   return type;
}
