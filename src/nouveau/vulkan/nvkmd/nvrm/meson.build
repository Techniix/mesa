# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT

# NVIDIA RM (nvrm) nvkmd backend, built as an internal static library.
# Talks to the NVIDIA kernel driver (e.g. /dev/graphics/nvidiaN on Haiku)
# rather than to DRM/libdrm.

nvkmd_nvrm_files = files(
  'nvkmd_nvrm.h',
  'nvkmd_nvrm_ctx.c',
  'nvkmd_nvrm_dev.c',
  'nvkmd_nvrm_mem.c',
  'nvkmd_nvrm_pdev.c',
  'nvkmd_nvrm_va.c',
  'nvkmd_nvrm_sync.c',
  'nvRmApi.c',
  'nvRmSemSurf.c',
)

# The NVIDIA open-gpu-kernel-modules checkout supplying the RM SDK headers
# (nv.h, nvRmApi.h, nvtypes.h, ctrl/...) lives outside this tree; its location
# is configured via -Dnvk-nvrm-source-path rather than a machine-local symlink.
nvrm_source_path = get_option('nvk-nvrm-source-path')
if nvrm_source_path == ''
  error('The nvrm nvkmd backend requires -Dnvk-nvrm-source-path to point at a ' +
        'checkout of NVIDIA open-gpu-kernel-modules.')
endif
fs = import('fs')
if not fs.is_absolute(nvrm_source_path)
  nvrm_source_path = meson.global_source_root() / nvrm_source_path
endif
if not fs.is_dir(nvrm_source_path / 'src/common/sdk/nvidia/inc')
  error('nvk-nvrm-source-path (@0@) does not look like an open-gpu-kernel-modules checkout'
        .format(nvrm_source_path))
endif

# The SDK header roots sit under an external path, so they can't go through
# include_directories() (which rejects absolute source-tree dirs). Feed them as
# -I compile args instead, and re-export the same args through the dependency
# below so consumers of nvkmd_nvrm.h (notably nvkmd.c in libnvk) can resolve the
# transitively-included SDK headers.
nvrm_sdk_args = []
foreach subdir : ['src/nvidia/arch/nvalloc/common/inc',
                  'src/nvidia/arch/nvalloc/unix/include',
                  'src/common/sdk/nvidia/inc',
                  'src/nvidia/inc/kernel']
  nvrm_sdk_args += '-I' + (nvrm_source_path / subdir)
endforeach

# The expected NVRM version is compiled in and checked against the running
# kernel module at device-enumeration time (see nvkmd_nvrm_pdev.c).
nvrm_version_args = [
  '-DNVKMD_NVRM_VERSION="' + get_option('nvk-nvrm-version') + '"',
]

nvkmd_nvrm_inc = [
  include_directories('.'),       # nvrm/: nvkmd_nvrm.h, nvRmApi.h
  include_directories('../..'),   # the vulkan/ dir: nvkmd/nvkmd.h, nvk_*.h
]

libnvkmd_nvrm = static_library(
  'nvkmd_nvrm',
  nvkmd_nvrm_files,
  include_directories : [
    inc_include,
    inc_src,
    inc_nouveau_winsys,
    nvkmd_nvrm_inc,
  ],
  dependencies : nvk_deps,
  c_args : [no_override_init_args, nvk_flags, nvrm_sdk_args, nvrm_version_args],
  gnu_symbol_visibility : 'hidden',
)

idep_nvkmd_nvrm = declare_dependency(
  link_with : libnvkmd_nvrm,
  include_directories : nvkmd_nvrm_inc,
  compile_args : nvrm_sdk_args,
)
