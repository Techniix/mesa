/*
 * Copyright © 2026 The Quasar authors
 * SPDX-License-Identifier: MIT
 *
 * Renga presentation through a bitmap hook; see wsi_common_renga.h and
 * Quasar's docs/RENGA-WSI.md.
 *
 * Frame path
 * ----------
 * Swapchain images are ordinary optimal-tiling images the application
 * renders to. Each has a linear, host-visible "blit buffer" (Mesa's
 * WSI_SWAPCHAIN_BUFFER_BLIT with CPU image parameters): Mesa's common present
 * code records the copy from the image into that buffer in the application's
 * own present submission, on the application's queue, and signals the image's
 * present fence (chain->base.fences[i]). Each blit buffer is wrapped once in a
 * BBitmap over the area that backs its mapping, so handing a frame to the
 * window costs no CPU copy.
 *
 * A present thread waits on the present fence -- never the application's
 * thread -- and gives the newest finished frame to the hook. The hook gives
 * back the bitmap it showed before, whose image becomes free for acquire.
 *
 * Image states
 * ------------
 *   FREE      idle on the GPU and not on screen: can be acquired
 *   ACQUIRED  owned by the application
 *   QUEUED    presented; the present thread has not handed it over yet
 *   SHOWN     its bitmap is the hook's current bitmap
 * Only FREE images are handed out by acquire, so an image is never rendered
 * to while its blit or app_server may still read it (that was the tearing
 * fixed in VideoStreamsWsi by rotating three bitmaps). Mesa's common code
 * then signals the application's acquire semaphore or fence with an
 * already-signalled payload, which is correct because the image is idle.
 *
 * Present modes
 * -------------
 * MAILBOX and IMMEDIATE: the newest finished frame is shown and older queued
 * ones are freed. FIFO: frames are shown in order, one per retrace when the
 * application has a BApplication and the screen reports retraces, otherwise
 * in order as fast as they finish. Real vblank pacing comes with Renga's GPU
 * presentation path (F-47); this path has no vblank feedback from app_server.
 *
 * Bitmap ownership (matches VideoStreamsWsi, which applications rely on)
 * ------------------------------------------------------------------------
 * The bitmap the hook currently shows belongs to the hook. Whatever
 * SetBitmap() returns comes back to the swapchain; a bitmap that is not one
 * of ours (left by an earlier swapchain, or the window's first bitmap) is
 * deleted. Clients delete the bitmap returned by SetBitmap(NULL) when they
 * tear a window down.
 *
 * On destruction the swapchain's memory is freed, so the bitmap on screen
 * must not keep pointing at it. The last frame is copied once into a plain
 * BBitmap that owns its memory, and that is left with the hook.
 */

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <new>
#include <thread>

#include <Application.h>
#include <Bitmap.h>
#include <OS.h>
#include <Screen.h>

#include "util/u_debug.h"
#include "vk_util.h"
#include "wsi_common_renga.h"


/* The hook interface. Must match the declaration applications compile
 * against (VideoStreamsWsi's Wsi.cpp, Zink's hgl_sw_winsys.h): one virtual
 * destructor, then GetSize, then SetBitmap. */
class BitmapHook {
public:
   virtual ~BitmapHook() {};
   virtual void GetSize(uint32_t &width, uint32_t &height) = 0;
   virtual BBitmap *SetBitmap(BBitmap *bmp) = 0;
};


namespace {

enum class ImageState { FREE, ACQUIRED, QUEUED, SHOWN };

struct renga_image {
   struct wsi_image base;
   BBitmap *bitmap;
   ImageState state;
};

struct renga_swapchain {
   struct wsi_swapchain base;

   BitmapHook *hook;
   VkExtent2D extent;
   bool debug;

   std::mutex lock;
   std::condition_variable changed;   /* any image state, or quit */
   std::deque<uint32_t> queued;       /* QUEUED images, oldest first */
   bool quit;
   std::thread present_thread;

   struct renga_image *images;        /* base.image_count entries */
};

} /* namespace */


const struct wsi_renga_swapchain_bitmap_hook *
wsi_renga_find_bitmap_hook(const VkSwapchainCreateInfoKHR *create_info)
{
   for (const VkBaseInStructure *s =
           (const VkBaseInStructure *)create_info->pNext;
        s != NULL; s = s->pNext) {
      if (s->sType == WSI_RENGA_STRUCTURE_TYPE_SWAPCHAIN_BITMAP_HOOK)
         return (const struct wsi_renga_swapchain_bitmap_hook *)s;
   }
   return NULL;
}


static struct renga_swapchain *
renga_chain(struct wsi_swapchain *chain)
{
   return container_of(chain, struct renga_swapchain, base);
}


static struct wsi_image *
renga_get_wsi_image(struct wsi_swapchain *wsi_chain, uint32_t image_index)
{
   return &renga_chain(wsi_chain)->images[image_index].base;
}


/* The extent the window wants now, or the swapchain's when it cannot say. */
static VkExtent2D
renga_hook_extent(struct renga_swapchain *chain)
{
   uint32_t width = 0, height = 0;
   chain->hook->GetSize(width, height);
   if (width == 0 || height == 0)
      return chain->extent;
   return (VkExtent2D) { width, height };
}


static VkResult
renga_check_suboptimal(struct renga_swapchain *chain)
{
   VkExtent2D want = renga_hook_extent(chain);
   if (want.width != chain->extent.width || want.height != chain->extent.height)
      return VK_SUBOPTIMAL_KHR;
   return VK_SUCCESS;
}


static VkResult
renga_acquire_next_image(struct wsi_swapchain *wsi_chain,
                         const VkAcquireNextImageInfoKHR *info,
                         uint32_t *image_index)
{
   struct renga_swapchain *chain = renga_chain(wsi_chain);
   std::unique_lock<std::mutex> lock(chain->lock);

   auto find_free = [chain]() -> int {
      for (uint32_t i = 0; i < chain->base.image_count; i++) {
         if (chain->images[i].state == ImageState::FREE)
            return (int)i;
      }
      return -1;
   };

   int index = find_free();
   if (index < 0) {
      if (info->timeout == 0)
         return VK_NOT_READY;
      /* UINT64_MAX means no timeout; wait_for with a huge duration would
       * overflow the clock, so treat anything over a day as forever. */
      const uint64_t day_ns = 86400ull * 1000000000ull;
      if (info->timeout >= day_ns) {
         chain->changed.wait(lock, [&] {
            return (index = find_free()) >= 0 || chain->quit;
         });
      } else if (!chain->changed.wait_for(lock,
                    std::chrono::nanoseconds(info->timeout), [&] {
                       return (index = find_free()) >= 0 || chain->quit;
                    })) {
         return VK_TIMEOUT;
      }
      if (index < 0)
         return VK_ERROR_OUT_OF_DATE_KHR;
   }

   chain->images[index].state = ImageState::ACQUIRED;
   *image_index = (uint32_t)index;
   lock.unlock();
   return renga_check_suboptimal(chain);
}


/* VK_EXT_swapchain_maintenance1: the application gives back acquired
 * images it will not present. */
static VkResult
renga_release_images(struct wsi_swapchain *wsi_chain,
                     uint32_t count, const uint32_t *indices)
{
   struct renga_swapchain *chain = renga_chain(wsi_chain);
   std::lock_guard<std::mutex> lock(chain->lock);
   for (uint32_t i = 0; i < count; i++) {
      assert(indices[i] < chain->base.image_count);
      if (chain->images[indices[i]].state == ImageState::ACQUIRED)
         chain->images[indices[i]].state = ImageState::FREE;
   }
   chain->changed.notify_all();
   return VK_SUCCESS;
}


/* Called by Mesa's common code in the application's thread, right after it
 * submitted the blit and the present fence. Only records the image. */
static VkResult
renga_queue_present(struct wsi_swapchain *wsi_chain, uint32_t image_index,
                    uint64_t present_id, const VkPresentRegionKHR *damage)
{
   struct renga_swapchain *chain = renga_chain(wsi_chain);
   (void)present_id;
   (void)damage;

   {
      std::lock_guard<std::mutex> lock(chain->lock);
      assert(chain->images[image_index].state == ImageState::ACQUIRED);
      chain->images[image_index].state = ImageState::QUEUED;
      chain->queued.push_back(image_index);
   }
   chain->changed.notify_all();
   return renga_check_suboptimal(chain);
}


/* Show image `index`: hand its bitmap to the hook and take back the one that
 * was shown. Called with chain->lock held. */
static void
renga_show(struct renga_swapchain *chain, uint32_t index)
{
   chain->images[index].state = ImageState::SHOWN;
   BBitmap *old = chain->hook->SetBitmap(chain->images[index].bitmap);
   if (old == NULL || old == chain->images[index].bitmap)
      return;
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->images[i].bitmap == old) {
         chain->images[i].state = ImageState::FREE;
         return;
      }
   }
   /* Not ours: an earlier swapchain's last frame or the window's own. */
   delete old;
}


static bool
renga_fence_done(struct renga_swapchain *chain, uint32_t index)
{
   VkFence fence = chain->base.fences[index];
   if (fence == VK_NULL_HANDLE)
      return true;
   const struct wsi_device *wsi = chain->base.wsi;
   return wsi->WaitForFences(chain->base.device, 1, &fence, VK_TRUE,
                             UINT64_MAX) == VK_SUCCESS;
}


static void
renga_wait_retrace()
{
   if (be_app == NULL)
      return;
   BScreen screen(B_MAIN_SCREEN_ID);
   /* 50 ms: never stall presentation for long if retraces stop. */
   screen.WaitForRetrace(50000);
}


static void
renga_present_thread(struct renga_swapchain *chain)
{
   std::unique_lock<std::mutex> lock(chain->lock);
   for (;;) {
      chain->changed.wait(lock, [chain] {
         return chain->quit || !chain->queued.empty();
      });
      if (chain->quit)
         return;

      bool fifo = chain->base.present_mode == VK_PRESENT_MODE_FIFO_KHR ||
                  chain->base.present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR;

      uint32_t index;
      if (fifo) {
         index = chain->queued.front();
         chain->queued.pop_front();
      } else {
         /* Newest wins; older queued frames are dropped once their blits
          * have finished, so they are idle when they become FREE. */
         index = chain->queued.back();
         chain->queued.pop_back();
      }
      std::deque<uint32_t> dropped;
      if (!fifo)
         dropped.swap(chain->queued);

      /* The waits happen without the lock, so acquire and present in the
       * application's threads are never blocked behind the GPU. */
      lock.unlock();
      for (uint32_t d : dropped)
         renga_fence_done(chain, d);
      bool ok = renga_fence_done(chain, index);
      if (fifo)
         renga_wait_retrace();
      lock.lock();

      for (uint32_t d : dropped)
         chain->images[d].state = ImageState::FREE;
      if (ok) {
         renga_show(chain, index);
      } else {
         /* Device lost or similar: do not show a frame that may be
          * incomplete, and let the application see the error at its next
          * submission. */
         chain->images[index].state = ImageState::FREE;
      }
      chain->changed.notify_all();
   }
}


static VkResult
renga_wait_for_present(struct wsi_swapchain *wsi_chain, uint64_t value,
                       uint64_t timeout)
{
   /* Mesa's common code signals the present-id timeline when the present
    * submission completes. With this path the frame reaches the hook right
    * after that, so the semaphore is the right signal. */
   return wsi_swapchain_wait_for_present_semaphore(wsi_chain, value, timeout);
}


/* Give the hook a copy of the frame on screen that owns its memory, since
 * the swapchain's memory is about to be freed. Called with chain->lock held
 * and the present thread stopped. */
static void
renga_detach_from_screen(struct renga_swapchain *chain)
{
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      struct renga_image *image = &chain->images[i];
      if (image->state != ImageState::SHOWN)
         continue;

      BBitmap *copy = new (std::nothrow) BBitmap(image->bitmap->Bounds(), 0,
         image->bitmap->ColorSpace());
      if (copy != NULL && copy->InitCheck() == B_OK) {
         copy->ImportBits(image->bitmap);
      } else {
         delete copy;
         copy = NULL;
      }
      BBitmap *old = chain->hook->SetBitmap(copy);
      if (old != NULL && old != image->bitmap)
         delete old;
      image->state = ImageState::FREE;
   }
}


static VkResult
renga_destroy(struct wsi_swapchain *wsi_chain,
              const VkAllocationCallbacks *allocator)
{
   struct renga_swapchain *chain = renga_chain(wsi_chain);

   {
      std::lock_guard<std::mutex> lock(chain->lock);
      chain->quit = true;
   }
   chain->changed.notify_all();
   if (chain->present_thread.joinable())
      chain->present_thread.join();

   {
      std::lock_guard<std::mutex> lock(chain->lock);
      renga_detach_from_screen(chain);
   }

   /* Presents still in flight read the images and write the blit buffers
    * the bitmaps wrap. vkDestroySwapchainKHR may be called with them
    * pending, so wait here before freeing anything. */
   for (uint32_t i = 0; i < chain->base.image_count; i++)
      renga_fence_done(chain, i);

   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      delete chain->images[i].bitmap;
      if (chain->images[i].base.image != VK_NULL_HANDLE)
         wsi_destroy_image(&chain->base, &chain->images[i].base);
   }

   vk_free(allocator, chain->images);
   wsi_swapchain_finish(&chain->base);
   chain->~renga_swapchain();
   vk_free(allocator, chain);
   return VK_SUCCESS;
}


/* A BBitmap over the blit buffer's mapping. The mapping comes from an area
 * (RM user mappings on Haiku are areas); B_BITMAP_IS_AREA lets app_server
 * clone that area instead of copying the pixels. */
static BBitmap *
renga_create_bitmap(struct renga_swapchain *chain, struct wsi_image *image)
{
   if (image->cpu_map == NULL)
      return NULL;
   area_id area = area_for(image->cpu_map);
   area_info info;
   if (area < 0 || get_area_info(area, &info) != B_OK)
      return NULL;
   ptrdiff_t offset = (uint8_t *)image->cpu_map - (uint8_t *)info.address;

   BBitmap *bitmap = new (std::nothrow) BBitmap(area, offset,
      BRect(0, 0, chain->extent.width - 1, chain->extent.height - 1),
      B_BITMAP_IS_AREA, B_RGB32, image->row_pitches[0]);
   if (bitmap != NULL && bitmap->InitCheck() != B_OK) {
      delete bitmap;
      bitmap = NULL;
   }
   return bitmap;
}


VkResult
wsi_renga_create_swapchain(VkDevice device,
                           struct wsi_device *wsi_device,
                           const VkSwapchainCreateInfoKHR *create_info,
                           const VkAllocationCallbacks *allocator,
                           void *bitmap_hook,
                           struct wsi_swapchain **swapchain_out)
{
   /* B_RGB32 is BGRA in memory; other formats would need a conversion the
    * hook cannot do. The surface reports only these formats. */
   if (create_info->imageFormat != VK_FORMAT_B8G8R8A8_UNORM &&
       create_info->imageFormat != VK_FORMAT_B8G8R8A8_SRGB)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   void *memory = vk_zalloc(allocator, sizeof(struct renga_swapchain), 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (memory == NULL)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct renga_swapchain *chain = new (memory) renga_swapchain();

   chain->hook = (BitmapHook *)bitmap_hook;
   chain->extent = create_info->imageExtent;
   chain->debug = debug_get_bool_option("RENGA_WSI_DEBUG", false);

   /* CPU image parameters without alloc_shm: the blit buffer is ordinary
    * host-visible Vulkan memory that Mesa maps. (alloc_shm would import host
    * memory, VK_EXT_external_memory_host, which the RM back end cannot do
    * yet; Quasar R-5.) */
   struct wsi_cpu_image_params cpu_params = {
      .base = { .image_type = WSI_IMAGE_TYPE_CPU },
      .alloc_shm = NULL,
   };

   VkResult result = wsi_swapchain_init(wsi_device, &chain->base, device,
                                        create_info, &cpu_params.base,
                                        allocator);
   if (result != VK_SUCCESS) {
      chain->~renga_swapchain();
      vk_free(allocator, chain);
      return result;
   }

   chain->base.destroy = renga_destroy;
   chain->base.get_wsi_image = renga_get_wsi_image;
   chain->base.acquire_next_image = renga_acquire_next_image;
   chain->base.release_images = renga_release_images;
   chain->base.queue_present = renga_queue_present;
   chain->base.wait_for_present = renga_wait_for_present;
   chain->base.present_mode =
      wsi_swapchain_get_present_mode(wsi_device, create_info);
   chain->base.image_count = create_info->minImageCount;

   chain->images = (struct renga_image *)vk_zalloc(allocator,
      sizeof(struct renga_image) * chain->base.image_count, 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (chain->images == NULL) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_finish;
   }

   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      result = wsi_create_image(&chain->base, &chain->base.image_info,
                                &chain->images[i].base);
      if (result != VK_SUCCESS)
         goto fail_images;
      chain->images[i].state = ImageState::FREE;
      chain->images[i].bitmap = renga_create_bitmap(chain,
                                                    &chain->images[i].base);
      if (chain->images[i].bitmap == NULL) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail_images;
      }
   }

   try {
      chain->present_thread = std::thread(renga_present_thread, chain);
   } catch (...) {
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto fail_images;
   }

   if (chain->debug) {
      fprintf(stderr, "wsi/renga: swapchain %ux%u, %u images, present mode %d\n",
              chain->extent.width, chain->extent.height,
              chain->base.image_count, chain->base.present_mode);
   }

   *swapchain_out = &chain->base;
   return VK_SUCCESS;

fail_images:
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      delete chain->images[i].bitmap;
      if (chain->images[i].base.image != VK_NULL_HANDLE)
         wsi_destroy_image(&chain->base, &chain->images[i].base);
   }
   vk_free(allocator, chain->images);
fail_finish:
   wsi_swapchain_finish(&chain->base);
   chain->~renga_swapchain();
   vk_free(allocator, chain);
   return result;
}
