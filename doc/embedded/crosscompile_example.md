## Crosscompile mlpack example for an embedded hardware

In this article, we explore how to add mlpack to a CMake project that cross-compiles code to embedded hardware.  See also these related other guides, which may be useful to read before this one:

 * [Run mlpack bindings on a Raspberry Pi](crosscompile_armv7.md)
 * [Set up cross-compilation toolchain for mlpack](supported_boards.md)

### Cloning mlpack example respository

mlpack has an [example repository](https://github.com/mlpack/examples) that
shows a number of applications and use cases for mlpack, including embedded
deployment.  In this tutorial, we are interested in the `embedded/` directory,
which provides a CMake project template that compiles a random forest
application to embedded hardware.  This project template can be adapted to a new
project, or its pieces can be incorporated into an existing CMake project.

The first step is to clone the examples repository:

```sh
git clone git@github.com:mlpack/examples.git
```

Next, let's look at the `CMakeLists.txt` (e.g. the CMake configuration) in the
`embedded/crosscompile_random_forest/` directory.

### Analysing CMakeLists.txt

The first part of the code, printed below or [available here](https://github.com/mlpack/examples/blob/master/embedded/crosscompile_random_forest/CMakeLists.txt),
defines the project name and includes two useful CMake configuration files:

 * `Autodownload.cmake`: downloads mlpack's dependencies from defined locations
 * `ConfigureCrossCompile.cmake`: set up CMake configuration for cross-compilation
 
The next steps set the required C++ standard to C++17 (which is the minimum required
version for mlpack), and enable OpenMP for parallelism.

```cmake
cmake_minimum_required(VERSION 3.6)
project(RandomForest)

include(CMake/Autodownload.cmake)
include(CMake/ConfigureCrossCompile.cmake)

option(USE_OPENMP "If available, use OpenMP for parallelization." ON)

# Set required standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If we're using gcc, then we need to link against pthreads to use std::thread,
# which we do in the tests.
if (CMAKE_COMPILER_IS_GNUCC)
  find_package(Threads)
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()
```

#### Downloading dependencies

Once we have included the configs and defined the C++ standard, then we need to
download these dependencies. In the following, we have added the link
for each one of them, but feel free to adapt the link to different versions or
locations.

Once `get_deps` has downloaded and extracted each one of them, the next step
would be to append the include directories for these libraries to the
`MLPACK_INCLUDE_DIRS` list variable. In the case of Armadillo, it could be set
in header-only mode or a linkable library thus these two variable exist in this
specific case. However, the `MLPACK_LIBRARIES` variable is not necessary if you
have defined armadillo in header-only mode.

```c++
search_openblas(0.3.26)

get_deps(https://files.mlpack.org/armadillo-12.6.5.tar.gz armadillo armadillo-12.6.5.tar.gz)
set(ARMADILLO_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
find_package(Armadillo REQUIRED)
# Include directories for the previous dependencies.
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})

# Find stb_image.h and stb_image_write.h.
get_deps(https://mlpack.org/files/stb.tar.gz stb stb.tar.gz)
set(STB_IMAGE_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${STB_IMAGE_INCLUDE_DIR}")

# Find ensmallen.
get_deps(https://www.ensmallen.org/files/ensmallen-latest.tar.gz ensmallen ensmallen-latest.tar.gz)
set(ENSMALLEN_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")

# Find cereal.
get_deps(https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.0.tar.gz cereal cereal-1.3.0.tar.gz)
set(CEREAL_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})
```

#### Setting up include directories and source files

The last part of our CMake file consists of merging all of the above components
together. First we are going to find the OpenMP package if the user defined the
variable above, then we will start including the directories of our
program. You will need to do the same for your software if you are trying to
integrate this example into an existing development, this should be done by
defining an include variable using `set` directive or add them directly in
`target_include_directories`.
Regarding the source code, a similar process by either adding then to
`add_executables` or by appending the source files to `SOURCES_FILES` variable. 

Finally do not forget to add any external library that you need to link against
in `target_link_libraries`.

```c++
# Detect OpenMP support in a compiler. If the compiler supports OpenMP, flags
# to compile with OpenMP are returned and added.  Note that MSVC does not
# support a new-enough version of OpenMP to be useful.
if (USE_OPENMP)
  find_package(OpenMP)
endif ()

if (OpenMP_FOUND AND OpenMP_CXX_VERSION VERSION_GREATER_EQUAL 3.0.0)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
else ()
  # Disable warnings for all the unknown OpenMP pragmas.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set(OpenMP_CXX_FLAGS "")
endif ()

include_directories(BEFORE ${MLPACK_INCLUDE_DIRS})
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src/)

## User includes go here

# Finally, add any cross-compilation support libraries (they may need to come
# last).  If we are not cross-compiling, no changes will happen here.
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CROSS_COMPILE_SUPPORT_LIBRARIES})

## Add your source files to SOURCES_FILES list

add_executable(RandomForest main.cpp ${SOURCES_FILES})
target_sources(RandomForest PRIVATE ${SOURCE_FILES})

## Do not forget to add the include variables in here.

target_include_directories(RandomForest PRIVATE
  ${MLPACK_INCLUDE_DIRS}
  /path/to/mlpack/src/
  /path/to/your/own/include
)

## Do not forget to add libraries that you are linking against in here.

target_link_libraries(RandomForest PRIVATE -static
  ${MLPACK_LIBRARIES}
)
```

#### Optimization and crosscompilation

If you are interested in adding specific compiler flags to optimize operations
on your hardware. Please feel free to look at mlpack
`CMake/crosscompile-arch-flags.cmake` you can either integrate the variable by
copy and paste them directly to this CMake file or you can
copy the entire `crosscompile-arch-config.cmake` file from `mlpack/CMake` and included
it close to `Autodownload.cmake` at start of this tutorial but in this case you need
to specify the `BOARD_NAME` variable as we did in the this [tutorial](crosscompile_armv7.md).

Once you have added all the source files and the headers for your applications,
you can create your own build directory and build the software using `cmake`,
and then `make`. Your cmake command should be similar to the following:

```sh
cmake \
    -DBUILD_TESTS=ON \
    -DBOARD_NAME=(Check below) \
    -DCMAKE_CROSSCOMPILING=ON \
    -DCMAKE_TOOLCHAIN_FILE=../CMake/crosscompile-toolchain.cmake \
    -DTOOLCHAIN_PREFIX=(Check below) \
    -DCMAKE_SYSROOT=(Check below) \
```

In order to fill the `TOOLCHAIN_PREFIX` or the `CMAKE_SYSROOT`, or if
you are interested in a different compiler toolchain please refer to the
following [table](supported_boards.md).

