/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVKMD_NVRM_LOG_H
#define NVKMD_NVRM_LOG_H 1

#include "vk_log.h"

#include "nvstatus.h"

/* Human-readable description + symbolic name for an NV_STATUS, derived from the
 * SDK's own X-macro table so it stays in sync with the codes RM actually
 * returns. Used purely for log messages -- a "0x1a (Ran out of a critical
 * resource, other than memory [NV_ERR_INSUFFICIENT_RESOURCES])" beats a bare
 * hex code when triaging a failed RM call.
 *
 * nvstatuscodes.h carries an include guard (SDK_NVSTATUSCODES_H, already pulled
 * in transitively via nvstatus.h), so undefine it before re-including -- the
 * same trick the SDK's own nvstatus.c uses to build its table. */
static inline const char *
nvkmd_nvrm_status_str(NV_STATUS status)
{
   switch (status) {
#undef NV_STATUS_CODE
#undef SDK_NVSTATUSCODES_H
#define NV_STATUS_CODE(name, code, str) case name: return str " [" #name "]";
#include "nvstatuscodes.h"
#undef NV_STATUS_CODE
   default:
      return "Unknown error code!";
   }
}

/* Best-effort NV_STATUS -> VkResult mapping. RM exposes far more failure modes
 * than Vulkan has error codes, so this only distinguishes the cases NVK can act
 * on differently (out-of-memory vs. device-lost vs. unsupported); everything
 * else collapses to VK_ERROR_UNKNOWN. Callers that need a specific result for a
 * known case (e.g. INVALID_ARGUMENT -> INCOMPATIBLE_DRIVER during probe) should
 * still special-case it themselves. */
static inline VkResult
nvkmd_nvrm_status_to_vk(NV_STATUS status)
{
   switch (status) {
   case NV_OK:
      return VK_SUCCESS;
   case NV_ERR_NO_MEMORY:
   case NV_ERR_INSUFFICIENT_RESOURCES:
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   case NV_ERR_NOT_SUPPORTED:
      return VK_ERROR_FEATURE_NOT_PRESENT;
   case NV_ERR_GPU_IS_LOST:
   case NV_ERR_RESET_REQUIRED:
   case NV_ERR_TIMEOUT:
      return VK_ERROR_DEVICE_LOST;
   default:
      return VK_ERROR_UNKNOWN;
   }
}

/* Log a failed RM call through the Vulkan debug-utils path (so validation
 * layers / vkconfig see it) and evaluate to the mapped VkResult, e.g.
 *   return nvkmd_nvrm_error(log_obj, nvRes, "nvRmApiAlloc(virtual memory)");
 * `what` should name the failed operation. */
#define nvkmd_nvrm_error(log_obj, nvRes, what)                              \
   vk_errorf((log_obj), nvkmd_nvrm_status_to_vk(nvRes),                     \
             "%s failed: 0x%08x (%s)", (what), (unsigned)(nvRes),           \
             nvkmd_nvrm_status_str(nvRes))

/* goto-style check for functions that unwind to an `error:` label and keep the
 * pending result in `vkRes` (the pattern in nvkmd_nvrm_pdev.c / _ctx.c). The
 * call expression is stringified into the log message, so the failing RM call
 * is identified verbatim. */
#define NVRM_CHECK(call)                                                    \
   do {                                                                     \
      NV_STATUS _nvRes = (call);                                            \
      if (_nvRes != NV_OK) {                                               \
         vkRes = nvkmd_nvrm_error(log_obj, _nvRes, #call);                  \
         goto error;                                                        \
      }                                                                     \
   } while (0)

#endif /* NVKMD_NVRM_LOG_H */
