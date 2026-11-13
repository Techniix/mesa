#include "nvRmApi.h"

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __HAIKU__
#include <sys/ioccom.h>
#else
#include <sys/ioctl.h>
#include <sys/mman.h>
#endif

#include "nv_escape.h"
#include "nv-unix-nvos-params-wrappers.h"
#include "nvstatus.h"
#ifdef __HAIKU__
#include "nv-haiku.h"
#endif


static int nvRmIoctl(int fd, NvU32 cmd, void *pParams, NvU32 paramsSize)
{
	int res;
	do {
#ifdef __HAIKU__
		res = ioctl(fd, cmd + NV_HAIKU_BASE, pParams, paramsSize);
#else
		/* Matches the NVIDIA Linux RM ABI: _IOWR(NV_IOCTL_MAGIC, cmd, <params>),
		 * i.e. an in/out ioctl carrying paramsSize bytes. */
		res = ioctl(fd, _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, cmd, paramsSize), pParams);
#endif
		if (res < 0) {
			res = errno;
		}
	} while ((res == EINTR || res == EAGAIN));
	return res;
}


NvU32 nvRmApiAlloc(NvRmApi *api, NvU32 hParent, NvU32 *hObject, NvU32 hClass, void *pAllocParams)
{
	NVOS21_PARAMETERS p = {
		.hRoot = api->hClient,
		.hObjectParent = hParent,
		.hObjectNew = *hObject,
		.hClass = hClass,
		.pAllocParms = pAllocParams
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_ALLOC, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	*hObject = p.hObjectNew;
	return p.status;
}

NvU32 nvRmApiAllocOsDescriptor(NvRmApi *api, NvU32 hParent, NvU32 *hObject, void *address, NvU64 size, bool writable)
{
	NVOS32_PARAMETERS p = {
		.hRoot = api->hClient,
		.hObjectParent = hParent,
		.function = NVOS32_FUNCTION_ALLOC_OS_DESCRIPTOR,
	};
	p.data.AllocOsDesc.type = NVOS32_TYPE_IMAGE;
	/* MEMORY_HANDLE_PROVIDED is intentionally unset, so RM generates the
	 * handle and returns it in hMemory below. */
	p.data.AllocOsDesc.flags = NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED;
	/* Anonymous user memory is write-back cacheable and cannot be guaranteed
	 * contiguous, so the OS-descriptor path requires exactly these attrs. */
	p.data.AllocOsDesc.attr =
		DRF_DEF(OS32, _ATTR, _LOCATION, _PCI) |
		DRF_DEF(OS32, _ATTR, _COHERENCY, _WRITE_BACK) |
		DRF_DEF(OS32, _ATTR, _PHYSICALITY, _NONCONTIGUOUS) |
		DRF_DEF(OS32, _ATTR, _PAGE_SIZE, _4KB);
	p.data.AllocOsDesc.attr2 =
		DRF_DEF(OS32, _ATTR2, _GPU_CACHEABLE, _NO) |
		(writable
			? DRF_DEF(OS32, _ATTR2, _PROTECTION_USER, _READ_WRITE)
			: DRF_DEF(OS32, _ATTR2, _PROTECTION_USER, _READ_ONLY));
	p.data.AllocOsDesc.descriptor = (NvP64)(NvUPtr)address;
	p.data.AllocOsDesc.limit = size - 1;
	p.data.AllocOsDesc.descriptorType = NVOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS;

	int ret = nvRmIoctl(api->fd, NV_ESC_RM_VID_HEAP_CONTROL, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	*hObject = p.data.AllocOsDesc.hMemory;
	return p.status;
}

NvU32 nvRmApiFree(NvRmApi *api, NvU32 hObject)
{
	if (hObject == 0) {
		return NV_OK;
	}
	NVOS00_PARAMETERS p = {
		.hRoot = api->hClient,
		.hObjectOld = hObject
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_FREE, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return p.status;
}

NvU32 nvRmApiControl(NvRmApi *api, NvU32 hObject, NvU32 cmd, void *pParams, NvU32 paramsSize)
{
	NVOS54_PARAMETERS p = {
		.hClient = api->hClient,
		.hObject = hObject,
		.cmd = cmd,
		.params = pParams,
		.paramsSize = paramsSize
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_CONTROL, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return p.status;
}

NvU32 nvRmApiMapMemoryDma(NvRmApi *api, NvU32 hDevice, NvU32 hDma, NvU32 hMemory, NvU64 offset, NvU64 length, NvU32 flags, NvU64 *dmaOffset)
{
	NVOS46_PARAMETERS p = {
		.hClient = api->hClient,
		.hDevice = hDevice,
		.hDma = hDma,
		.hMemory = hMemory,
		.offset = offset,
		.length = length,
		.flags = flags,
		.dmaOffset = *dmaOffset,
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_MAP_MEMORY_DMA, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	*dmaOffset = p.dmaOffset;
	return p.status;
}

NvU32 nvRmApiUnmapMemoryDma(NvRmApi *api, NvU32 hDevice, NvU32 hDma, NvU32 hMemory, NvU32 flags, NvU64 dmaOffset)
{
	NVOS47_PARAMETERS p = {
		.hClient = api->hClient,
		.hDevice = hDevice,
		.hDma = hDma,
		.hMemory = hMemory,
		.flags = flags,
		.dmaOffset = dmaOffset
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_UNMAP_MEMORY_DMA, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return p.status;
}

NvU32 nvRmApiMapMemory(NvRmApi *api, NvU32 hDevice, NvU32 hMemory, NvU64 offset, NvU64 length, bool isSysMem, NvU32 flags, NvRmApiMapping *mapping)
{
	const char *nodeName;
	if (isSysMem) {
		nodeName = NVRM_CTL_NODE_NAME;
	} else {
		nodeName = api->nodeName;
	}

	mapping->address = NULL;
	int memFd = open(nodeName, O_RDWR | O_CLOEXEC);
	if (memFd < 0) {
		return NV_ERR_GENERIC;
	}

	nv_ioctl_nvos33_parameters_with_fd p = {
		.params = {
			.hClient = api->hClient,
			.hDevice = hDevice,
			.hMemory = hMemory,
			.offset = offset,
			.length = length,
			.pLinearAddress = 0,
			.flags = flags
		},
		.fd = memFd
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_MAP_MEMORY, &p, sizeof(p));
	if (ret < 0) {
		p.params.status = NV_ERR_GENERIC;
		goto done1;
	}
	if (p.params.status != NV_OK) {
		goto done1;
	}
	mapping->stubLinearAddress = p.params.pLinearAddress;

#ifdef __HAIKU__
	nv_haiku_map_params mapParams = {
		.addressSpec = B_ANY_ADDRESS,
		.protection = B_READ_AREA | B_WRITE_AREA,
	};
	if (isSysMem) {
		strcpy(mapParams.name, "NVRM sysmem");
		mapParams.protection |= B_CLONEABLE_AREA;
	} else {
		strcpy(mapParams.name, "NVRM vidmem");
	}
	ret = nvRmIoctl(memFd, NV_HAIKU_MAP, &mapParams, sizeof(mapParams));
	if (ret < 0) {
		p.params.status = NV_ERR_GENERIC;
		goto done1;
	}
	mapping->area = ret;
	mapping->address = mapParams.address;
#else
	mapping->address = (void*)mmap(0, length, PROT_READ|PROT_WRITE, MAP_SHARED, memFd, 0);
	if (mapping->address == MAP_FAILED) {
		p.params.status = NV_ERR_GENERIC;
		goto done1;
	}
	mapping->size = length;
#endif

done1:
	close(memFd);
	return p.params.status;
}

NvU32 nvRmApiUnmapMemory(NvRmApi *api, NvU32 hDevice, NvU32 hMemory, NvU32 flags, NvRmApiMapping *mapping)
{
	if (mapping->address == NULL) {
		return NV_OK;
	}
#ifdef __HAIKU__
	delete_area(mapping->area);
#else
	munmap(mapping->address, mapping->size);
#endif

	NVOS34_PARAMETERS p = {
		.hClient = api->hClient,
		.hDevice = hDevice,
		.hMemory = hMemory,
		.pLinearAddress = mapping->stubLinearAddress,
		.status = 0,
		.flags = flags,
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_RM_UNMAP_MEMORY, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return p.status;
}

NvU32 nvRmApiRegisterFd(NvRmApi *api, int ctlFd)
{
	nv_ioctl_register_fd_t p = {
		.ctl_fd = ctlFd,
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_REGISTER_FD, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return NV_OK;
}

NvU32 nvRmApiAllocOsEvent(NvRmApi *api, int fd)
{
	nv_ioctl_alloc_os_event_t p = {
		.hClient = api->hClient,
		.fd = fd
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_ALLOC_OS_EVENT, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return p.Status;
}

NvU32 nvRmApiFreeOsEvent(NvRmApi *api, int fd)
{
	nv_ioctl_free_os_event_t p = {
		.hClient = api->hClient,
		.fd = fd
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_FREE_OS_EVENT, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return p.Status;
}

NvU32 nvRmApiCardInfo(NvRmApi *api, nv_ioctl_card_info_t *ci, size_t size)
{
	int ret = nvRmIoctl(api->fd, NV_ESC_CARD_INFO, ci, size);
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	return NV_OK;
}

NvU32 nvRmApiGetVersion(NvRmApi *api, char *versionOut, size_t versionSize)
{
	nv_ioctl_rm_api_version_t p = {
		.cmd = NV_RM_API_VERSION_CMD_QUERY,
	};
	int ret = nvRmIoctl(api->fd, NV_ESC_CHECK_VERSION_STR, &p, sizeof(p));
	if (ret < 0) {
		return NV_ERR_GENERIC;
	}
	if (p.reply != NV_RM_API_VERSION_REPLY_RECOGNIZED) {
		return NV_ERR_NOT_SUPPORTED;
	}
	p.versionString[NV_RM_API_VERSION_STRING_LENGTH - 1] = '\0';
	snprintf(versionOut, versionSize, "%s", p.versionString);
	return NV_OK;
}
