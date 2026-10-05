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
