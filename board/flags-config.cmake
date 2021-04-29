# This function provides a set of specific flags for each supported board
# Depending on the processor type. The objective is to optimize for size.
# Thus, all of the fllowing flags are chosen carefully to reduce binary 
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

option(RPI0 "Optimize compiler flags for Raspberry PI 0." OFF)
option(RPI1 "Optimize compiler flags for Raspberry PI 1." OFF)
option(RPI2 "Optimize compiler flags for Raspberry PI 2." OFF)
option(RPI3 "Optimize compiler flags for Raspberry PI 3." OFF)
option(RPI4 "Optimize compiler flags for Raspberry PI 4." OFF)
option(BV "Optimize compiler flags for Beagleboard V." OFF)
option(KATAMI "Optimize compiler flags for Pentium 3 Katami processors." OFF)
option(COPPERMINE "Optimize compiler flags for Pentium 3 Coppermine processors." OFF)
option(NORTHWOOD "Optimize compiler flags for Pentium 4 Northwood processors." OFF)

# Set specific platforms CMAKE CXX flags.
if(RPI0 OR RPI1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=arm1176jzf-s")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "ARMV6")
  set(OPENBLAS_BINARY "32")
elseif(RPI2)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a7")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "ARMV7")
  set(OPENBLAS_BINARY "32")
elseif(RPI3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a53")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "CORTEXA53")
  set(OPENBLAS_BINARY "64")
elseif(RPI4)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=cortex-a72")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "CORTEXA72")
  set(OPENBLAS_BINARY "64")
elseif(BV)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "RISCV64_GENERIC")
  set(OPENBLAS_BINARY "64") 
elseif(KATAMI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=pentium3")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "KATAMI")
  set(OPENBLAS_BINARY "32")
elseif(COPPERMINE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=pentium3")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "COPPERMINE")
  set(OPENBLAS_BINARY "32")
elseif(NORTHWOOD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=pentium4")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  set(OPENBLAS_TARGET "NORTHWOOD")
  set(OPENBLAS_BINARY "32")
endif()
