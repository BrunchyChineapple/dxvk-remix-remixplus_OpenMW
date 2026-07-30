subdir('unit')

# Note: on arm building only the unit tests
if target_machine.cpu() != 'armv8'

dxvkrt_test_root = meson.global_source_root().replace('\\', '/') + '/tests/rtx/dxvk_rt_testing/'
# Tests for the meson.build, not merely the directory. subdir() needs the build file, and the directory
# exists without it: the build installs test artefacts into dxvk_rt_testing/apics, so a tree that never had
# these tests checked out still grows the directory on its first build -- and every reconfigure after that
# failed with "Nonexistent build file". fs.is_dir was the wrong predicate for what this guard protects.
if fs.is_file('dxvk_rt_testing/meson.build')
  subdir('dxvk_rt_testing')
endif

subdir('apps/RemixAPI')
subdir('apps/RemixAPI_C')
subdir('apps/HydraTestRender')

if dxvk_is_ninja
  # apps that are compiled as a part of dxvk-remix
  dxvkrt_output_targets += {
    'apics/RemixAPI'    : RemixAPI_exepath,
    'apics/RemixAPI_C'  : RemixAPI_C_exepath,
    'apics/HydraTestRender'  : HydraTestRender_exepath
  }
endif

endif
