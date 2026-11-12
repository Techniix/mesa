/*
 * Copyright © 2026 The Quasar authors
 * SPDX-License-Identifier: MIT
 *
 * Renga presentation for headless surfaces.
 *
 * On Renga, applications attach a Vulkan surface to a window through a
 * "bitmap hook": an object with GetSize() and SetBitmap(BBitmap*) that shows
 * the bitmap it is given in a BView. Zink, OBS Studio and Eglantine create
 * a VK_EXT_headless_surface and hand their hook to the Renga surface layer
 * (Quasar's RengaWsi), which passes it down here in the swapchain create
 * info. A headless swapchain created without it behaves as Mesa's headless
 * swapchain always has.
 *
 * The struct below is private between that layer and Mesa. It is not a
 * Vulkan extension, its sType is outside the Khronos ranges, and it may
 * change with the layer; see Quasar's docs/RENGA-WSI.md.
 */

#ifndef WSI_COMMON_RENGA_H
#define WSI_COMMON_RENGA_H

#include "wsi_common_private.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 'RGA' 0x02. Khronos sTypes are below 0x7fffffff and allocated from
 * 1000000000 in blocks of 1000; this value is in neither range. */
#define WSI_RENGA_STRUCTURE_TYPE_SWAPCHAIN_BITMAP_HOOK ((VkStructureType)0x52474102)

struct wsi_renga_swapchain_bitmap_hook {
   VkStructureType sType;
   const void *pNext;
   /* A BitmapHook* (C++), valid for the swapchain's lifetime. */
   void *bitmap_hook;
};

const struct wsi_renga_swapchain_bitmap_hook *
wsi_renga_find_bitmap_hook(const VkSwapchainCreateInfoKHR *create_info);

VkResult
wsi_renga_create_swapchain(VkDevice device,
                           struct wsi_device *wsi_device,
                           const VkSwapchainCreateInfoKHR *create_info,
                           const VkAllocationCallbacks *allocator,
                           void *bitmap_hook,
                           struct wsi_swapchain **swapchain_out);

#ifdef __cplusplus
}
#endif

#endif /* WSI_COMMON_RENGA_H */
