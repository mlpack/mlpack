# Flags for each supported board

function(flags BOARD_NAME)
  # Set all minimization flags for all platforms
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fdata-sections -ffunction-sections -fomit-frame-pointer -fno-unwind-tables -fno-asynchronous-unwind-tables -fvisibility=hidden -fshort-enums -finline-small-functions -findirect-inlining -fno-common") #-flto -fuse-ld=gold # There is an issue with gold link when compiling on Ubuntu 16

  # Set the CMAKE CXX flags
  if(BOARD_NAME STREQUAL "rpi0" OR BOARD_NAME STREQUAL "rpi1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfp -mfloat-abi=hard -march=armv6zk -mcpu=arm1176jzf-s" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  elseif(BOARD_NAME STREQUAL "rpi2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4 -mfloat-abi=hard̈́ -march=armv7-a -mtune=cortex-a7" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")  
  elseif(BOARD_NAME STREQUAL "rpi3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mcpu=cortex-a53 " PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  elseif(BOARD_NAME STREQUAL "rpi4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard -mcpu=cortex-a72 -mneon-for-64bits" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  elseif(BOARD_NAME STREQUAL "parallella")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} " PARENT_SCOPE)
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  elseif(BOARD_NAME STREQUAL "x86")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=generic32" PARENT_SCOPE)
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  elseif(BOARD_NAME STREQUAL "i386")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=i386" PARENT_SCOPE)
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
 elseif(BOARD_NAME STREQUAL "i486")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=i486" PARENT_SCOPE)
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
  endif()
endfunction()
