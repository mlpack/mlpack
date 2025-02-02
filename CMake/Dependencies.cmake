# This function finds dependencies for mlpack and provide the user for
# different cases to pull these dependencies, whether it is by finding the one
# located on the system or by using the autodownloader.

macro(find_OpenMP)
  if (USE_OPENMP)
    find_package(OpenMP)
  endif ()

  if (OpenMP_FOUND AND OpenMP_CXX_VERSION VERSION_GREATER_EQUAL 3.0.0)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
  else ()
    # Disable warnings for all the unknown OpenMP pragmas.
    if (NOT MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    else ()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
    endif ()
    set(OpenMP_CXX_FLAGS "")
  endif ()
endmacro()

macro(autodownload compile)
  set(version 0.3.28)

  find_package(BLAS PATHS ${CMAKE_BINARY_DIR})
  if (NOT BLAS_FOUND OR (NOT BLAS_LIBRARIES))
    get_deps(https://github.com/xianyi/OpenBLAS/releases/download/v${version}/OpenBLAS-${version}.tar.gz OpenBLAS OpenBLAS-${version}.tar.gz)
    if (NOT compile)
      message(WARNING "OpenBLAS is downloaded but not compiled. Please compile
      OpenBLAS before compiling mlpack")
    else()
      execute_process(COMMAND make NO_SHARED=1 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version})
      file(GLOB OPENBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${version}/libopenblas.a")
      set(BLAS_openblas_LIBRARY ${OPENBLAS_LIBRARIES})
      set(LAPACK_openblas_LIBRARY ${OPENBLAS_LIBRARIES})
      set(BLAS_FOUND ON)
    endif()
  endif()

  find_package(Armadillo "${ARMADILLO_VERSION}" PATHS ${CMAKE_BINARY_DIR})
  if (NOT ARMADILLO_FOUND)
    if (NOT CMAKE_CROSSCOMPILING)
      find_package(BLAS QUIET)
      find_package(LAPACK QUIET)
      if (NOT BLAS_FOUND AND NOT LAPACK_FOUND)
        message(FATAL_ERROR "Can not find BLAS or LAPACK!  These are required for Armadillo.  Please install one of them---or install Armadillo---before installing mlpack.")
      endif()
    endif()
    get_deps(https://files.mlpack.org/armadillo-12.6.5.tar.gz armadillo armadillo-12.6.5.tar.gz)
    set(ARMADILLO_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    if (NOT CMAKE_CROSSCOMPILING)
      find_package(Armadillo REQUIRED)
    endif()
    # Include directories for the previous dependencies.
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
  endif()

  ## Keep STB out of this until we merge the stb PR

  find_package(Ensmallen "${ENSMALLEN_VERSION}" PATHS ${CMAKE_BINARY_DIR})
  if (NOT ENSMALLEN_FOUND)
    get_deps(https://www.ensmallen.org/files/ensmallen-latest.tar.gz ensmallen ensmallen-latest.tar.gz)
    set(ENSMALLEN_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_package(Ensmallen REQUIRED)
    if (ENSMALLEN_FOUND)
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")
    endif()
  endif()

  find_package(cereal "${CEREAL_VERSION}" PATHS ${CMAKE_BINARY_DIR})
  if (NOT CEREAL_FOUND)
    get_deps(https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.0.tar.gz cereal cereal-1.3.0.tar.gz)
    set(CEREAL_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_package(cereal REQUIRED)
    if (CEREAL_FOUND)
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})
    endif()
  endif()

  find_OpenMP() 
endmacro()

# Download and compile OpenBLAS if we are cross compiling mlpack for a specific
# architecture. The function takes the version of OpenBLAS as variable.
macro(crosscompile)
  # 1. Pull openblas and crossocompile it
  search_openblas(0.3.28)
  # 2. Call autodownload, DONE
  autodownload(ON)
endmacro()
