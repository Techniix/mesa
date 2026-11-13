/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nouveau.h"

#include "util/macros.h"

#include <xf86drm.h>

VkResult
nvkmd_nouveau_enum_pdev(struct vk_object_base *log_obj,
                        enum nvk_debug debug_flags,
                        nvkmd_enum_pdev_visitor visitor,
                        void *arg)
{
   /* libdrm returns a maximum of 256 devices (see MAX_DRM_NODES in libdrm) */
   drmDevicePtr devices[256];
   int max_devices = drmGetDevices2(0, devices, ARRAY_SIZE(devices));
   if (max_devices < 1)
      return VK_SUCCESS;

   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < (uint32_t)max_devices; i++) {
      struct nvkmd_pdev *pdev = NULL;
      result = nvkmd_nouveau_try_create_pdev(devices[i], log_obj,
                                             debug_flags, &pdev);
      /* Incompatible DRM device, skip. */
      if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
         result = VK_SUCCESS;
         continue;
      }
      if (result != VK_SUCCESS)
         break;

      /* The visitor takes ownership of pdev (and destroys it on failure). */
      result = visitor(pdev, arg);
      if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
         result = VK_SUCCESS;
         continue;
      }
      if (result != VK_SUCCESS)
         break;
   }

   drmFreeDevices(devices, max_devices);
   return result;
}
