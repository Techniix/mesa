# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT

# nouveau nvkmd backend, built as an internal static library.
# Talks to the kernel through DRM/libdrm and the nouveau winsys.

nvkmd_nouveau_files = files(
  'nvkmd_nouveau.h',
  'nvkmd_nouveau.c',
  'nvkmd_nouveau_ctx.c',
  'nvkmd_nouveau_dev.c',
  'nvkmd_nouveau_mem.c',
  'nvkmd_nouveau_pdev.c',
  'nvkmd_nouveau_va.c',
)

libnvkmd_nouveau = static_library(
  'nvkmd_nouveau',
  nvkmd_nouveau_files,
  include_directories : [
    inc_include,
    inc_src,
    inc_nouveau_winsys,
    inc_nouveau_drm,
    include_directories('../..'),  # the vulkan/ dir: nvkmd/nvkmd.h, nvk_*.h
  ],
  dependencies : [nvk_deps, dep_libdrm, idep_nouveau_ws],
  c_args : [no_override_init_args, nvk_flags],
  gnu_symbol_visibility : 'hidden',
)

idep_nvkmd_nouveau = declare_dependency(
  link_with : libnvkmd_nouveau,
  include_directories : include_directories('.'),
)
