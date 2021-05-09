## This file handles cross-compilation configurations for aarch64,
## known as arm64. The objective of this file is to find and assign
## cross-compiler and the entire toolchain.
##
## This configuration works best with the buildroot toolchain.  When using this
## file, be sure to set the TOOLCHAIN_PREFIX and CMAKE_SYSROOT variables,
## preferably via the CMake configuration command (e.g. `-DCMAKE_SYSROOT=<...>`).
##
## Currently, we recommend using buildroot toolchain for
## cross-compilation. Here is the link to download the toolchains:
## https://toolchains.bootlin.com/

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSROOT)
set(TOOLCHAIN_PREFIX "" CACHE STRING "Path for toolchain for cross compiler and other compilation tools.")

# Ensure that CMake tries to build static libraries when testing the compiler.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(CMAKE_AR "${TOOLCHAIN_PREFIX}gcc-ar" CACHE FILEPATH "" FORCE)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_LINKER ${TOOLCHAIN_PREFIX}ld)
set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
set(CMAKE_C_ARCHIVE_FINISH  true)
set(CMAKE_FORTRAN_COMPILER ${TOOLCHAIN_PREFIX}gfortran)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy CACHE INTERNAL "objcopy tool")
set(CMAKE_SIZE_UTIL ${TOOLCHAIN_PREFIX}size CACHE INTERNAL "size tool")

## Here are the standard ROOT_PATH if you are using the standard toolchain
## if you are using a different toolchain you have to specify that too.
set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --sysroot=${CMAKE_SYSROOT}" CACHE INTERNAL "" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
