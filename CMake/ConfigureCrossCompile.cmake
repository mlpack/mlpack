# This file adds the necessary configurations to cross compile
# mlpack for embedded systems. You need to set the following variables
# from the command line: CMAKE_SYSROOT and TOOLCHAIN_PREFIX.
# This file will compile OpenBLAS if it is downloaded and it is not
# available on your system in order to find the BLAS library.  If OpenBLAS will
# be compiled, the OPENBLAS_TARGET variable must be set.  This can be done
# by, e.g., setting BOARD_NAME (which will set OPENBLAS_TARGET in
# `board/flags-config.cmake`).

if (CMAKE_CROSSCOMPILING)
  include(board/flags-config.cmake)
  if (NOT CMAKE_SYSROOT AND (NOT TOOLCHAIN_PREFIX))
    message(FATAL_ERROR "Neither CMAKE_SYSROOT nor TOOLCHAIN_PREFIX are set; please set both of them and try again.")
  elseif(NOT CMAKE_SYSROOT)
    message(FATAL_ERROR "Cannot configure: CMAKE_SYSROOT must be set when performing cross-compiling!")
  elseif(NOT TOOLCHAIN_PREFIX)
    message(FATAL_ERROR "Cannot configure: TOOLCHAIN_PREFIX must be set when performing cross-compiling!")
  endif()
endif()

macro(search_openblas version)
  set(BLA_STATIC ON)
  find_package(BLAS)
  if (NOT BLAS_FOUND OR (NOT BLAS_LIBRARIES))
    if(NOT OPENBLAS_TARGET)
      message(FATAL_ERROR "Cannot compile OpenBLAS: OPENBLAS_TARGET is not set.  Either set that variable, or set BOARD_NAME correctly!")
    endif()
    get_deps(https://github.com/xianyi/OpenBLAS/releases/download/v${version}/OpenBLAS-${version}.tar.gz OpenBLAS OpenBLAS-${version}.tar.gz)
    if (NOT MSVC)
      if (NOT EXISTS "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/libopenblas.a")
        execute_process(COMMAND make TARGET=${OPENBLAS_TARGET} BINARY=${OPENBLAS_BINARY} HOSTCC=gcc CC=${CMAKE_C_COMPILER} FC=${CMAKE_FORTRAN_COMPILER} NO_SHARED=1
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version})
      endif()
      file(GLOB OPENBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/libopenblas.a")
      set(BLAS_openblas_LIBRARY ${OPENBLAS_LIBRARIES})
      set(LAPACK_openblas_LIBRARY ${OPENBLAS_LIBRARIES}) 
      set(BLA_VENDOR OpenBLAS)
      set(BLAS_FOUND ON)
    endif()
  endif()
  find_library(GFORTRAN NAMES libgfortran.a)
  find_library(PTHREAD NAMES libpthread.a)
  set(CROSS_COMPILE_SUPPORT_LIBRARIES ${GFORTRAN} ${PTHREAD})
endmacro()
