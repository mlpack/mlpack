# This file adds the necessary configurations to cross compile
# mlpack for embedded systems. You need to set the following variables
# from the command line: the CMAKE_SYSROOT, TOOLCHAIN_PREFIX and the
# board type.
# This file will compile OpenBLAS if it is downloaded and it is not
# available on your system in order to find the BLAS library.

if (CMAKE_CROSSCOMPILING)
  include(board/flags-config.cmake)
  if (NOT CMAKE_SYSROOT AND (NOT TOOLCHAIN_PREFIX))
    message(FATAL_ERROR "Neither of CMAKE_SYSROOT or TOOLCHAIN_PREFIX is set, please set both of them and try again")
  elseif(NOT CMAKE_SYSROOT)
    message(FATAL_ERROR "Can not proceed CMAKE_SYSROOT is not set")
  elseif(NOT TOOLCHAIN_PREFIX)
    message(FATAL_ERROR "Cant not proceed TOOLCHAIN_PREFIX is not set")
  endif()
endif()

macro(search_openblas version)
  set(BLA_STATIC ON)
  find_package(BLAS)
  if (NOT BLAS_FOUND OR (NOT BLAS_LIBRARIES))
    if(NOT OPENBLAS_TARGET)
      message(FATAL_ERROR "Cant not proceed, OPENBLAS_TARGET is not set, and to either set that or BOARD_NAME")
    endif()
    get_deps(https://github.com/xianyi/OpenBLAS/releases/download/v${version}/OpenBLAS-${version}.tar.gz OpenBLAS OpenBLAS-${version}.tar.gz)
    if (NOT MSVC)
      if (NOT EXISTS "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/libopenblas.a")
        execute_process(COMMAND make TARGET=${OPENBLAS_TARGET} BINARY=${OPENBLAS_BINARY} HOSTCC=gcc CC=${CMAKE_C_COMPILER} FC=${CMAKE_FORTRAN_COMPILER} NO_SHARED=1
                        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version})
      endif()
      file(GLOB OPENBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/libopenblas.a")
      # set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
      # set(LAPACK_LIBRARIES ${OPENBLAS_LIBRARIES})
      set(BLAS_openblas_LIBRARY ${OPENBLAS_LIBRARIES})
      set(LAPACK_openblas_LIBRARY ${OPENBLAS_LIBRARIES})  
      message(STATUS "SHOW BLAS libraries: ${BLAS_LIBRARIES}")
      set(BLA_VENDOR OpenBLAS)
      set(BLAS_FOUND ON)
    endif()
  endif()
  find_library(GFORTRAN NAMES libgfortran.a)
  find_library(PTHREAD NAMES libpthread.a)
  set(COMPILER_SUPPORT_LIBRARIES ${COMPILER_SUPPORT_LIBRARIES} ${GFORTRAN} ${PTHREAD})
  message(STATUS "SHOW BLAS libraries 2: ${BLAS_LIBRARIES}")
endmacro()
