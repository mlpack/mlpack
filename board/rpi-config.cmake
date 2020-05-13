## A small sketch to see where we can go with RPI configurations

function(RPI BOARD_NAME)
if(BOARD_NAME STREQUAL "rpi0" OR BOARD_NAME STREQUAL "rpi1")
  set(CMAKE_SYSTEM_PROCESSOR BCM2835)
elseif(BOARD_NAME STREQUAL "rpi2")
  set(CMAKE_SYSTEM_PROCESSOR BCM2836)
elseif(BOARD_NAME STREQUAL "rpi3")
  set(CMAKE_SYSTEM_PROCESSOR BCM2837)
elseif(BOARD_NAME STREQUAL "rpi4")
  set(CMAKE_SYSTEM_PROCESSOR BCM2711)
endif()

# Set a toolchain path.
set(TOOLCHAIN_PATH "")

if(BOARD_NAME STREQUAL "rpi0" OR BOARD_NAME STREQUAL "rpi1" OR BOARD_NAME STREQUAL "rpi2")
  set(CMAKE_CXX_COMPILER ${TC_PATH}arm-linux-gnueabihf-g++)
elseif(BOARD_NAME STREQUAL "rpi3" OR BOARD_NAME STREQUAL "rpi4")
  set(CMAKE_CXX_COMPILER ${TC_PATH}aarch64-linux-gnu-g++)
endif()

# Set all minimization flags for all platforms
set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -0s -flto ")

# Set the CMAKE CXX flags
if(BOARD_NAME STREQUAL "rpi0" OR BOARD_NAME STREQUAL "rpi1")
  set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfp -mfloat-abi=hard -march=armv6zk -mcpu=arm1176jzf-s")
elseif(BOARD_NAME STREQUAL "rpi2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4 -mfloat-abi=hard̈́ -march=armv7-a -mtune=cortex-a7")
elseif(BOARD_NAME STREQUAL "rpi3")
  set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard -mcpu=cortex-a53 -mneon-for-64bits")
elseif(BOARD_NAME STREQUAL "rpi4")
  set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard -mcpu=cortex-a72 -mneon-for-64bits")
endif()
endfunction()
