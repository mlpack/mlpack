# This file adds the necessary configurations to cross compile
# mlpack for embedded systems. You need to set the following variables
# from the command line: CMAKE_SYSROOT and TOOLCHAIN_PREFIX.
# This file will compile OpenBLAS if it is downloaded and it is not
# available on your system in order to find the BLAS library.  If OpenBLAS will
# be compiled, the OPENBLAS_TARGET variable must be set.  This can be done
# by, e.g., setting ARCH_NAME (which will set OPENBLAS_TARGET in
# `flags-config.cmake`).

if (CMAKE_CROSSCOMPILING)
  include(CMake/crosscompile-arch-config.cmake)
  if (NOT CMAKE_SYSROOT AND (NOT TOOLCHAIN_PREFIX))
    message(FATAL_ERROR "Neither CMAKE_SYSROOT nor TOOLCHAIN_PREFIX are set; please set both of them and try again.")
  elseif(NOT CMAKE_SYSROOT)
    message(FATAL_ERROR "Cannot configure: CMAKE_SYSROOT must be set when performing cross-compiling!")
  elseif(NOT TOOLCHAIN_PREFIX)
    message(FATAL_ERROR "Cannot configure: TOOLCHAIN_PREFIX must be set when performing cross-compiling!")
  endif()

  # Now make sure that we can still compile a simple test program.
  # (This ensures we didn't add any bad CXXFLAGS.)
  # Note that OUTPUT_VARIABLE is only available in newer versions of CMake!
  # CMake 3.23 (silently) introduced the variable.
  include(CheckCXXSourceCompiles)
  if (CMAKE_VERSION VERSION_LESS "3.22.0")
    check_cxx_source_compiles("int main() { return 0; }" COMPILE_SUCCESS)
    if (NOT COMPILE_SUCCESS)
      message(FATAL_ERROR "The C++ cross-compiler at ${CMAKE_CXX_COMPILER} is "
        "not able to compile a trivial test program.  Check the CXXFLAGS!")
    endif ()
  else ()
    check_cxx_source_compiles("int main() { return 0; }" COMPILE_SUCCESS
        OUTPUT_VARIABLE COMPILE_OUTPUT)
    if (NOT COMPILE_SUCCESS)
      message(FATAL_ERROR "The C++ cross-compiler at ${CMAKE_CXX_COMPILER} is "
        "not able to compile a trivial test program.  Compiler output:\n\n"
        "${COMPILE_OUTPUT}")
    endif ()
  endif ()
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
        set(OLD_CC $ENV{CC})
        set(ENV{COMMON_OPT} "${CMAKE_OPENBLAS_FLAGS}") # Pass our flags to OpenBLAS
        set(ENV{CC} "${CMAKE_C_COMPILER}")
        if (CCACHE_PROGRAM)
          set (ENV{CC} "${CCACHE_PROGRAM} $ENV{CC}")
        endif ()
        set(ENV{TARGET} "${OPENBLAS_TARGET}")
        set(ENV{BINARY} "${OPENBLAS_BINARY}")
        set(ENV{HOSTCC} "gcc")
        set(ENV{NO_SHARED} 1)
        # Set any extra environment variables that the user specified.
        # First, turn OPENBLAS_EXTRA_ARGS into a list.
        separate_arguments(ARG_LIST NATIVE_COMMAND ${OPENBLAS_EXTRA_ARGS})
        foreach (ARG ${ARG_LIST})
          # ARG will be of the form VAR=value; split on the =.
          string(REPLACE "=" ";" ARG_LIST ${ARG})
          list(LENGTH ${ARG_LIST} ARG_LEN)
          if (ARG_LEN EQUAL 2)
            list(GET ARG_LIST 0 ARG_NAME)
            list(GET ARG_LIST 1 ARG_VAL)
            set(ENV{${ARG_NAME}} ${ARG_VAL})
          else ()
            message(WARNING "OpenBLAS flag '${ARG}' is not of the form VAR=val;
ignored!")
          endif ()
        endforeach()

        # Now run the OpenBLAS build with all of the environment variables set.
        execute_process(
            COMMAND make
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version})

        # Don't leak the environment variables back into the rest of the
        # configuration.
        unset(ENV{NO_SHARED})
        unset(ENV{HOSTCC})
        unset(ENV{BINARY})
        unset(ENV{TARGET})
        unset(ENV{COMMON_OPT})
        set(ENV{CC} ${OLD_CC})
        foreach (ARG ${ARG_LIST})
          string(REPLACE "=" ";" ARG_LIST ${ARG})
          list(LENGTH ${ARG_LIST} ARG_LEN)
          if (ARG_LEN EQUAL 2)
            list(GET ARG_LIST 0 ARG_NAME)
            unset(ENV{${ARG_NAME}})
          endif ()
        endforeach ()
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
