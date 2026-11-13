/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nvkmd_nvrm.h"
#include "nvkmd_nvrm_log.h"

#include "vk_log.h"

#include "util/u_memory.h"
#include "util/hash_table.h"

#include "class/cl003e.h" // NV01_MEMORY_SYSTEM
#include "class/cl0040.h" // NV01_MEMORY_LOCAL_USER

#include "ctrl/ctrl0000/ctrl0000unix.h" // NV0000_CTRL_CMD_OS_UNIX_{EXPORT,IMPORT}_OBJECT_*
#include "ctrl/ctrl0000/ctrl0000client.h" // NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE
#include "ctrl/ctrl0041.h" // NV0041_CTRL_CMD_GET_SURFACE_INFO

#include <fcntl.h>
#include <unistd.h>


static VkResult
create_mem_or_close_bo(struct nvkmd_nvrm_dev *dev,
                       struct vk_object_base *log_obj,
                       enum nvkmd_mem_flags mem_flags,
                       bool isSystemMem,
                       NvHandle hMemoryPhys, uint64_t size_B,
                       enum nvkmd_va_flags va_flags,
                       uint8_t pte_kind, uint64_t va_align_B,
                       struct nvkmd_mem **mem_out)
{
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);
   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   VkResult result;

   struct nvkmd_nvrm_mem *mem = CALLOC_STRUCT(nvkmd_nvrm_mem);
   if (mem == NULL) {
      result = vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_bo;
   }

   nvkmd_mem_init(&dev->base, &mem->base, &nvkmd_nvrm_mem_ops,
                  mem_flags, size_B, dev->base.pdev->bind_align_B);
   mem->hMemoryPhys = hMemoryPhys;
   mem->isSystemMem = isSystemMem;

   result = nvkmd_dev_alloc_va(&dev->base, log_obj,
                               va_flags, pte_kind,
                               size_B, va_align_B,
                               0 /* fixed_addr */,
                               &mem->base.va);
   if (result != VK_SUCCESS)
      goto fail_mem;

   result = nvkmd_va_bind_mem(mem->base.va, log_obj, 0 /* va_offset_B */,
                              &mem->base, 0 /* mem_offset_B */, size_B);
   if (result != VK_SUCCESS)
      goto fail_va;

   *mem_out = &mem->base;

   return VK_SUCCESS;

fail_va:
   nvkmd_va_free(mem->base.va);
fail_mem:
   FREE(mem);
fail_bo:
   nvRmApiFree(&rm, hMemoryPhys);

   return result;
}

VkResult
nvkmd_nvrm_alloc_mem(struct nvkmd_dev *dev,
                        struct vk_object_base *log_obj,
                        uint64_t size_B, uint64_t align_B,
                        enum nvkmd_mem_flags flags,
                        struct nvkmd_mem **mem_out)
{
   return nvkmd_nvrm_alloc_tiled_mem(dev, log_obj, size_B, align_B,
                                        0 /* pte_kind */, 0 /* tile_mode */,
                                        flags, mem_out);
}

VkResult
nvkmd_nvrm_alloc_tiled_mem(struct nvkmd_dev *_dev,
                              struct vk_object_base *log_obj,
                              uint64_t size_B, uint64_t align_B,
                              uint8_t pte_kind, uint16_t tile_mode,
                              enum nvkmd_mem_flags flags,
                              struct nvkmd_mem **mem_out)
{
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   if (dev->base.pdev->debug_flags & NVK_DEBUG_FORCE_GART) {
      flags &= ~(NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM);
      flags |= NVKMD_MEM_GART;
   }
   if (flags & NVKMD_MEM_CAN_MAP) {
      flags &= ~(NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM);
      flags |= NVKMD_MEM_GART;
   }

   /* We assume that all discrete GPU maps are coherent: they're either cached
    * PCI sysmem (GART) or write-back VRAM maps. This matches the nouveau
    * backend and the import path in this file. NVK's SEPARATE query-pool
    * layout (used on every non-SOC GPU) relies on this flag to skip the
    * manual sync_{to,from}_gpu path; without it nvk_sync_queries_from_gpu and
    * nvk_ResetQueryPool assert. Tegra/SOC parts are not supported here. */
   if (_dev->pdev->dev_info.type != NV_DEVICE_TYPE_SOC)
      flags |= NVKMD_MEM_COHERENT;

   const uint32_t mem_align_B = _dev->pdev->bind_align_B;
   size_B = align64(size_B, mem_align_B);

   assert(util_is_power_of_two_or_zero64(align_B));
   const uint64_t va_align_B = MAX2(mem_align_B, align_B);

   enum nvkmd_va_flags va_flags = NVKMD_VA_GART;

	bool isSystemMem = (flags & NVKMD_MEM_GART) != 0;
	const NvU32 hClass = isSystemMem ?
		NV01_MEMORY_SYSTEM :
		NV01_MEMORY_LOCAL_USER;

	NV_MEMORY_ALLOCATION_PARAMS params = {
		.owner = pdev->hClient,
		.type = NVOS32_TYPE_IMAGE,
		.flags = (align_B != 0) ? NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE : 0,
		.attr = DRF_DEF(OS32, _ATTR, _PAGE_SIZE, _4KB),
		.size = size_B,
		.alignment = align_B,
	};
	if (isSystemMem) {
		params.attr |= DRF_DEF(OS32, _ATTR, _LOCATION, _PCI);
		params.attr |= DRF_DEF(OS32, _ATTR, _COHERENCY, _CACHED);
	} else {
		params.attr |= DRF_DEF(OS32, _ATTR, _LOCATION, _VIDMEM);
		params.attr |= DRF_DEF(OS32, _ATTR, _COHERENCY, _UNCACHED);
		params.flags |= NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM;
	}

	NvHandle hMemoryPhys = 0;
   NV_STATUS nvRes = nvRmApiAlloc(&rm, pdev->hDevice, &hMemoryPhys, hClass, &params);
   if (nvRes != NV_OK) {
      return vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "nvRmApiAlloc(%s) failed: 0x%08x (%s)",
                       isSystemMem ? "NV01_MEMORY_SYSTEM" : "NV01_MEMORY_LOCAL_USER",
                       (unsigned)nvRes, nvkmd_nvrm_status_str(nvRes));
   }

   return create_mem_or_close_bo(dev, log_obj, flags, isSystemMem,
                                 hMemoryPhys, size_B,
                                 va_flags, pte_kind, va_align_B,
                                 mem_out);
}

VkResult
nvkmd_nvrm_import_dma_buf(struct nvkmd_dev *_dev,
                             struct vk_object_base *log_obj,
                             int fd, struct nvkmd_mem **mem_out)
{
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS importParams = {
      .fd = fd,
      .object = {
         .type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM,
         .data.rmObject = {
            .hDevice = pdev->hDevice,
            .hParent = pdev->hDevice,
            .hObject = 0, /* allocated by RM */
         },
      },
   };
   NV_STATUS nvRes = nvRmApiControl(&rm, pdev->hClient,
                                    NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD,
                                    &importParams, sizeof(importParams));
   if (nvRes != NV_OK) {
      return vk_errorf(log_obj, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "OS_UNIX_IMPORT_OBJECT_FROM_FD failed: %#x", nvRes);
   }
   NvHandle hMemoryPhys = importParams.object.data.rmObject.hObject;

   /* Recover address space (sysmem vs vidmem) from the client. */
   NV0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS addrSpaceParams = {
      .hObject = hMemoryPhys,
   };
   nvRes = nvRmApiControl(&rm, pdev->hClient,
                          NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE,
                          &addrSpaceParams, sizeof(addrSpaceParams));
   if (nvRes != NV_OK) {
      nvRmApiFree(&rm, hMemoryPhys);
      return vk_errorf(log_obj, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "CLIENT_GET_ADDR_SPACE_TYPE failed: %#x", nvRes);
   }
   const bool isSystemMem =
      addrSpaceParams.addrSpaceType == NV0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_SYSMEM;

   /* Recover physical size from the memory object itself. */
   NV0041_CTRL_SURFACE_INFO sizeInfo = {
      .index = NV0041_CTRL_SURFACE_INFO_INDEX_PHYS_SIZE,
   };
   NV0041_CTRL_GET_SURFACE_INFO_PARAMS surfInfoParams = {
      .surfaceInfoListSize = 1,
      .surfaceInfoList = (NvP64)(uintptr_t)&sizeInfo,
   };
   nvRes = nvRmApiControl(&rm, hMemoryPhys,
                          NV0041_CTRL_CMD_GET_SURFACE_INFO,
                          &surfInfoParams, sizeof(surfInfoParams));
   if (nvRes != NV_OK) {
      nvRmApiFree(&rm, hMemoryPhys);
      return vk_errorf(log_obj, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "GET_SURFACE_INFO[PHYS_SIZE] failed: %#x", nvRes);
   }
   const uint64_t size_B =
      (uint64_t)sizeInfo.data * NV0041_CTRL_SURFACE_INFO_PHYS_SIZE_SCALE_FACTOR;

   /* All maps on these discrete GPUs are coherent: imported memory is either
    * CPU sysmem or write-back VRAM, matching the nouveau backend's assumption.
    * Host-visible memory types are HOST_COHERENT, so without this the
    * coherence assert in nvk_AllocateMemory would fire on re-import.
    */
   enum nvkmd_mem_flags mem_flags =
      NVKMD_MEM_SHARED | NVKMD_MEM_CAN_MAP | NVKMD_MEM_COHERENT;
   mem_flags |= isSystemMem ? NVKMD_MEM_GART : NVKMD_MEM_LOCAL;

   const enum nvkmd_va_flags va_flags = isSystemMem ? NVKMD_VA_GART : 0;
   const uint64_t va_align_B = dev->base.pdev->bind_align_B;

   return create_mem_or_close_bo(dev, log_obj, mem_flags, isSystemMem,
                                 hMemoryPhys, size_B,
                                 va_flags, 0 /* pte_kind */, va_align_B,
                                 mem_out);
}

VkResult
nvkmd_nvrm_import_userptr(struct nvkmd_dev *_dev,
                             struct vk_object_base *log_obj,
                             void *userptr, uint64_t size_B,
                             enum nvkmd_mem_flags flags,
                             struct nvkmd_mem **mem_out)
{
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   /* Host pointer imports are anonymous system memory, so they are always
    * GART and host-visible regardless of the requested placement. The kernel
    * locks exactly the pages spanning [userptr, userptr + size_B); size_B is
    * already page-aligned by the caller, so we pass it through unchanged
    * rather than inflating to bind_align_B (which would lock past the buffer).
    */
   flags &= ~(NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM);
   flags |= NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP;

   const uint32_t va_align_B = _dev->pdev->bind_align_B;

   NvHandle hMemoryPhys = 0;
   NV_STATUS nvRes = nvRmApiAllocOsDescriptor(&rm, pdev->hDevice, &hMemoryPhys,
                                              userptr, size_B, /*writable=*/true);
   if (nvRes != NV_OK) {
      return vk_errorf(log_obj, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "ALLOC_OS_DESCRIPTOR failed: %#x", nvRes);
   }

   VkResult result =
      create_mem_or_close_bo(dev, log_obj, flags, /*isSystemMem=*/true,
                             hMemoryPhys, size_B,
                             NVKMD_VA_GART, 0 /* pte_kind */, va_align_B,
                             mem_out);
   if (result != VK_SUCCESS)
      return result;

   nvkmd_nvrm_mem(*mem_out)->userptr = userptr;

   return VK_SUCCESS;
}

static void
nvkmd_nvrm_mem_free(struct nvkmd_mem *_mem)
{
   struct nvkmd_nvrm_mem *mem = nvkmd_nvrm_mem(_mem);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(mem->base.dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   nvkmd_va_free(mem->base.va);
   nvRmApiFree(&rm, mem->hMemoryPhys);
   FREE(mem);
}

static VkResult
nvkmd_nvrm_mem_map(struct nvkmd_mem *_mem,
                      struct vk_object_base *log_obj,
                      enum nvkmd_mem_map_flags map_flags,
                      void *fixed_addr,
                      void **map_out)
{
   struct nvkmd_nvrm_mem *mem = nvkmd_nvrm_mem(_mem);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_mem->dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   /* Host-pointer imports are already CPU-visible through the application's
    * own pointer; hand it straight back rather than asking RM for a second
    * mapping of the same physical pages. */
   if (mem->userptr != NULL) {
      *map_out = mem->userptr;
      return VK_SUCCESS;
   }

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   NvRmApiMapping *mapping = CALLOC_STRUCT(NvRmApiMapping);
   if (mapping == NULL)
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);

   NV_STATUS nvRes = nvRmApiMapMemory(&rm, pdev->hSubdevice, mem->hMemoryPhys, 0, mem->base.size_B, mem->isSystemMem, 0, mapping);
   if (nvRes != NV_OK) {
      free(mapping);
      return vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "nvRmApiMapMemory failed: 0x%08x (%s)",
                       (unsigned)nvRes, nvkmd_nvrm_status_str(nvRes));
   }

   if (_mesa_hash_table_insert(dev->mappings, mapping->address, mapping) == NULL) {
   	nvRmApiUnmapMemory(&rm, pdev->hSubdevice, mem->hMemoryPhys, 0, mapping);
      free(mapping);
      return vk_error(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   *map_out = mapping->address;
   return VK_SUCCESS;
}

static void
nvkmd_nvrm_mem_unmap(struct nvkmd_mem *_mem,
                        enum nvkmd_mem_map_flags flags,
                        void *map)
{
   struct nvkmd_nvrm_mem *mem = nvkmd_nvrm_mem(_mem);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_mem->dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   /* Host-pointer imports never created an RM mapping (see _mem_map). */
   if (mem->userptr != NULL)
      return;

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   /* Every address handed out by map() is recorded in dev->mappings, so a miss
    * is an internal invariant violation rather than a runtime error. */
   struct hash_entry *ent = _mesa_hash_table_search(dev->mappings, map);
   assert(ent != NULL);
   if (ent == NULL)
      return;
   NvRmApiMapping *mapping = ent->data;

   /* Drop the table entry before freeing its data so a later map() that reuses
    * this address can't observe a dangling pointer. */
   _mesa_hash_table_remove(dev->mappings, ent);

   /* Best-effort like nouveau's munmap: this is a void teardown op with no
    * caller to propagate a failure to. */
   nvRmApiUnmapMemory(&rm, pdev->hSubdevice, mem->hMemoryPhys, 0, mapping);

   free(mapping);
}

static VkResult
nvkmd_nvrm_mem_overmap(struct nvkmd_mem *_mem,
                          struct vk_object_base *log_obj,
                          enum nvkmd_mem_map_flags flags,
                          void *map)
{
   struct nvkmd_nvrm_mem *mem = nvkmd_nvrm_mem(_mem);

   return vk_errorf(log_obj, VK_ERROR_UNKNOWN,
                    "overmap unsupported (hMemoryPhys=%#x)", mem->hMemoryPhys);
}

static VkResult
nvkmd_nvrm_mem_export_dma_buf(struct nvkmd_mem *_mem,
                                 struct vk_object_base *log_obj,
                                 int *fd_out)
{
   struct nvkmd_nvrm_mem *mem = nvkmd_nvrm_mem(_mem);
   struct nvkmd_nvrm_dev *dev = nvkmd_nvrm_dev(_mem->dev);
   struct nvkmd_nvrm_pdev *pdev = nvkmd_nvrm_pdev(dev->base.pdev);

   struct NvRmApi rm;
   nvkmd_nvrm_dev_api_ctl(pdev, &rm);

   /* Fresh nvidiactl fd serves as the carrier; the kernel attaches the
    * exported handle to its nv_file_private. The fd may be sent across
    * processes via SCM_RIGHTS; closing it (in any process) drops the
    * export ref via rm_cleanup_file_private.
    */
   int carrier_fd = open(NVRM_CTL_NODE_NAME, O_RDWR | O_CLOEXEC);
   if (carrier_fd < 0) {
      return vk_errorf(log_obj, VK_ERROR_TOO_MANY_OBJECTS,
                       "open(%s) failed", NVRM_CTL_NODE_NAME);
   }

   NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS exportParams = {
      .fd = carrier_fd,
      .flags = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD_FALSE,
      .object = {
         .type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM,
         .data.rmObject = {
            .hDevice = pdev->hDevice,
            .hParent = pdev->hDevice,
            .hObject = mem->hMemoryPhys,
         },
      },
   };
   NV_STATUS nvRes = nvRmApiControl(&rm, pdev->hClient,
                                    NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD,
                                    &exportParams, sizeof(exportParams));
   if (nvRes != NV_OK) {
      close(carrier_fd);
      return vk_errorf(log_obj, VK_ERROR_TOO_MANY_OBJECTS,
                       "OS_UNIX_EXPORT_OBJECT_TO_FD failed: %#x", nvRes);
   }

   *fd_out = carrier_fd;
   return VK_SUCCESS;
}

static uint32_t
nvkmd_nvrm_mem_log_handle(struct nvkmd_mem *_mem)
{
   return 0;
}

const struct nvkmd_mem_ops nvkmd_nvrm_mem_ops = {
   .free = nvkmd_nvrm_mem_free,
   .map = nvkmd_nvrm_mem_map,
   .unmap = nvkmd_nvrm_mem_unmap,
   .overmap = nvkmd_nvrm_mem_overmap,
   .export_dma_buf = nvkmd_nvrm_mem_export_dma_buf,
   .log_handle = nvkmd_nvrm_mem_log_handle,
};
