# Copyright © 2021 Ilia Mirkin
# SPDX-License-Identifier: MIT

inc_nouveau_winsys=include_directories('winsys')

if with_nvk_nouveau_backend
  subdir('drm')      # defines inc_nouveau_drm
endif
subdir('headers')
if with_nvk_nouveau_backend
  subdir('winsys')   # defines idep_nouveau_ws
endif
if with_nouveau_vk
  subdir('rust')
  subdir('nil')
  subdir('compiler')
endif
if with_tools.contains('drm-shim')
  subdir('drm-shim')
endif

if with_nouveau_vk
  subdir('cubin')
  subdir('mme')
  subdir('vulkan')
endif
