# This function provides a set of specific flags for each supported board
# depending on the processor type. The objective is to optimize for size.
# Thus, all of the following flags are chosen carefully to reduce binary 
# footprints.

# Set generic minimization flags for all platforms.
# These flags are the same for all cross-compilation cases.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fdata-sections -ffunction-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer -fno-unwind-tables")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-asynchronous-unwind-tables -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fshort-enums -finline-small-functions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -findirect-inlining -fno-common") 
#-flto -fuse-ld=gold # There is an issue with gold link when compiling on 
# Ubuntu 16. At that point gcc linker did not integrate the flto support 
# inside and it was a separate plugin that need to be added. Therefore, 
# this can be added when mlpack Azure CI moves toward Ubuntu 20.

set(BOARD_NAME "" CACHE STRING "Specify Board name to optimize for.")
string(TOUPPER ${BOARD_NAME} BOARD)

# Set specific platforms CMAKE CXX flags.
if(BOARD MATCHES "RPI0" OR BOARD MATCHES "RPI1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=arm1176jzf-s")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "ARMV6")
  set(OPENBLAS_BINARY "32")
elseif(BOARD MATCHES "RPI2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a7")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "ARMV7")
  set(OPENBLAS_BINARY "32")
elseif(BOARD MATCHES "RPI3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a53")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "CORTEXA53")
  set(OPENBLAS_BINARY "64")
elseif(BOARD MATCHES "RPI4")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a72")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "CORTEXA72")
  set(OPENBLAS_BINARY "64")
elseif(BOARD MATCHES "BV")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "RISCV64_GENERIC")
  set(OPENBLAS_BINARY "64")
elseif(BOARD MATCHES "JETSONAGX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -matune=cortex-a76")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "ARM8")
  set(OPENBLAS_BINARY "64")
elseif(BOARD MATCHES "KATAMI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=pentium3")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "KATAMI")
  set(OPENBLAS_BINARY "32")
elseif(BOARD MATCHES "COPPERMINE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=pentium3")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "COPPERMINE")
  set(OPENBLAS_BINARY "32")
elseif(BOARD MATCHES "NORTHWOOD")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=pentium4")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "NORTHWOOD")
  set(OPENBLAS_BINARY "32")
elseif(BOARD)
  ## TODO: update documentation with a list of the supported boards.
  message(FATAL_ERROR "Given BOARD_NAME is not known; please choose a supported board from the list")
endif()
