## A small sketch to see where we can go with RPI configurations

## Not sure if we merge these with the arm files
## TODO:: VERIFY later

# if(BOARD_NAME STREQUAL "rpi0" OR BOARD_NAME STREQUAL "rpi1")
#   set(CMAKE_SYSTEM_PROCESSOR BCM2835)
# elseif(BOARD_NAME STREQUAL "rpi2")
#   set(CMAKE_SYSTEM_PROCESSOR BCM2836)
# elseif(BOARD_NAME STREQUAL "rpi3")
#   set(CMAKE_SYSTEM_NAME Linux)
#   set(CMAKE_SYSTEM_PROCESSOR BCM2837)
# elseif(BOARD_NAME STREQUAL "rpi4")
#   set(CMAKE_SYSTEM_PROCESSOR BCM2711)
# endif()

function(flags BOARD_NAME)
  # Set all minimization flags for all platforms
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -fomit-frame-pointer -fno-unwind-tables -fno-asynchronous-unwind-tables")

  # Set the CMAKE CXX flags
  if(BOARD_NAME STREQUAL "rpi0" OR BOARD_NAME STREQUAL "rpi1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfp -mfloat-abi=hard -march=armv6zk -mcpu=arm1176jzf-s" PARENT_SCOPE)
  elseif(BOARD_NAME STREQUAL "rpi2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4 -mfloat-abi=hard̈́ -march=armv7-a -mtune=cortex-a7" PARENT_SCOPE)
  elseif(BOARD_NAME STREQUAL "rpi3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -mcpu=cortex-a53 " PARENT_SCOPE)
  elseif(BOARD_NAME STREQUAL "rpi4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard -mcpu=cortex-a72 -mneon-for-64bits" PARENT_SCOPE)
  endif()
endfunction()
