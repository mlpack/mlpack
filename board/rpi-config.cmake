## A small sketch to see where we can go with RPI configurations

if(rpi0, 1)
  set(CMAKE_SYSTEM_PROCESSOR BCM2835)
elif(rpi 2)
  set(CMAKE_SYSTEM_PROCESSOR BCM2836)
elif(rpi 3)
  set(CMAKE_SYSTEM_PROCESSOR BCM2837)
elif(rpi 4)
  set(CMAKE_SYSTEM_PROCESSOR BCM2711)
endif()

# Set a toolchain path.
set(TOOLCHAIN_PATH "")

if(rpi0, 1, 2)
  set(CMAKE_CXX_COMPILER ${TC_PATH}arm-linux-gnueabihf-g++)
elif(rpi 3 4)
  set(CMAKE_CXX_COMPILER ${TC_PATH}aarch64-linux-gnu-g++)
endif()

# Set all minimization flags for all platforms
set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -0s -flto ")

# Set the CMAKE CXX flags
if(rpi0, 1)
  set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfp -mfloat-abi=hard -march=armv6zk -mcpu=arm1176jzf-s")
elif(rpi 2)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4 -mfloat-abi=hard̈́ -march=armv7-a -mtune=cortex-a7")
elif(rpi 3)
  set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard -mcpu=cortex-a53 -mneon-for-64bits")
elif(rpi 4)
  set(CMAKE_CX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-fp-armv8 -mfloat-abi=hard -mcpu=cortex-a72 -mneon-for-64bits")
endif()
