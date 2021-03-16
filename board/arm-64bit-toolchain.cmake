set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ARM)

if(MINGW OR CYGWIN OR WIN32)
    set(UTIL_SEARCH_CMD where)
elseif(UNIX OR APPLE)
    set(UTIL_SEARCH_CMD which)
endif()

## You need to add the full path in TOOLCHAIN_PREFIX, if you do not use the
## standard gcc toolchain 
set(TOOLCHAIN_PREFIX aarch64-linux-gnu-)

## In some distribution, a dynamic link for aarch64-linux-gnu-gcc may not be
## found or created, instead it might be labelled with the version at the end
## For instance: aarch64-linux-gnu-gcc-5
## Therefore, if dynamic link exists, you do not have to specify the version
set(VERSION_NUMBER "" CACHE STRING "Enter the version number of the compiler")

execute_process(
  COMMAND ${UTIL_SEARCH_CMD} ${TOOLCHAIN_PREFIX}gcc${VERSION_NUMBER}
  message("GCC VERSION IS ${VERSION_NUMBER}")
  OUTPUT_VARIABLE BINUTILS_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Without that flag CMake is not able to pass test compilation check
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_AR "${TOOLCHAIN_PREFIX}gcc-ar${VERSION_NUMBER}" CACHE FILEPATH "" FORCE)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc${VERSION_NUMBER})
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++${VERSION_NUMBER})
set(CMAKE_LINKER ${TOOLCHAIN_PREFIX}ld${VERSION_NUMBER})
set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_C_ARCHIVE_FINISH  true)
set(CMAKE_FORTRAN_COMPILER ${TOOLCHAIN_PREFIX}gfortran)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy${VERSION_NUMBER} CACHE INTERNAL "objcopy tool")
set(CMAKE_SIZE_UTIL ${TOOLCHAIN_PREFIX}size${VERSION_NUMBER} CACHE INTERNAL "size tool")

## There is no need to specify the CMAKE_SYSROOT if you are using the 
## standard toolchain. If you download you own toolchain you have to specify
## the path for sysroot as follows:
## set(CMAKE_SYSROOT /PathToToolchain/aarch64-buildroot-linux-gnu/sysroot)
## Or it can be specified from commandline
set(CMAKE_SYSROOT "" CACHE STRING "Enter path for sysroot")

## Here are the standard ROOT_PATH if you are using the standard toolchain
## if you are using a different toolchain you have to specify that too
set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} 
  /usr/aarch64-linux-gnu/ 
)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
