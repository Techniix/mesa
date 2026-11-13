/*
 * Copyright © 2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "nak_private.h"

/* The nouveau winsys / DRM bindings are only needed by the nouveau nvkmd
 * backend. Gate them on that backend rather than on HAVE_LIBDRM: libdrm may be
 * present on the host (e.g. a Linux build of the nvrm backend) without the
 * nouveau winsys being built, in which case nouveau_bo.h is unavailable. */
#ifdef NVK_NOUVEAU_WS
#include "nouveau_bo.h"
#include "nouveau_context.h"
#include "nouveau_device.h"

#include <xf86drm.h>
#include "drm-uapi/nouveau_drm.h"

#define DRM_RS_IOCTL(FOO) \
   DRM_RS_IOCTL_##FOO = DRM_IOCTL_##FOO

enum ENUM_PACKED drm_rs_ioctls {
   DRM_RS_IOCTL(NOUVEAU_EXEC),
};
#endif
