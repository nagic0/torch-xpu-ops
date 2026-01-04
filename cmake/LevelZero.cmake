option(USE_ZE_XPU "Build with Level Zero XPU support" ON)

if(DEFINED ENV{USE_ZE_XPU})
  set(USE_ZE_XPU $ENV{USE_ZE_XPU})
endif()

message(STATUS "USE_ZE_XPU is set to ${USE_ZE_XPU}")

if(NOT USE_ZE_XPU)
  return()
endif()

find_package(LevelZero)
if(NOT LevelZero_FOUND)
  message(FATAL_ERROR "Can NOT find LevelZero cmake helpers module!")
endif()

set(TORCH_XPU_OPS_LEVEL_ZERO_INCLUDE_DIR ${LevelZero_INCLUDE_DIR})

set(TORCH_XPU_OPS_LEVEL_ZERO_LIBRARIES ${LevelZero_LIBRARY})

list(PREPEND TORCH_XPU_OPS_LEVEL_ZERO_LIBRARIES "-Wl,--start-group")
list(APPEND TORCH_XPU_OPS_LEVEL_ZERO_LIBRARIES "-Wl,--end-group")
