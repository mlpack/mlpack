## Crosscompile mlpack example on an embedded hardware

In this article, we explore, how to crosscompile and run an mlpack example code
on an embedded hardware such as Raspberry Pi 2. In our previous documentations,
we have explored how to run mlpack bindings such as kNN command line program on
a Raspberry PI. Please refer to that article first and follow the first
part on how to Setup cross-compilation toolchain, and then continue with this
article.

mlpack has an example repository that demonstrates a set of examples showing how
to use the library source code on different dataset and usecases including
embedded deployment. This tutorial basically explain our necessary CMake configurations
that are required to integrate with your local CMake to download mlpack
dependencies and cross compile the entire software.

If you have not used mlpack example repsoitory, I highly recommend to clone
this repository from this link to follow this tutorial:

```sh
git clone git@github.com:mlpack/examples.git
```

You can explore this repository and see the available example, we are
interested in the `embedded/` directory. In this directory we are providing
a CMake template project and Random Forest example to use it on embedded
hardware. In this tutorial we are exploring the RandomForest example by first
analysing the CMakeLists.txt


In this part of code we are defining the project name and including two CMake
configurations files. The first one is the Autodowload function that is going
to pull the dependencies from respective locations that we are going to define.

The second configuration file is 

Then we are setting the C++ standard to use and define settings for `OpenMP` and
`std::thread`

```c++
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

Once we have included the configs and defined the started, then we need to pull
the dependencies from their locations, in the following we have added the link
for each one of them, but feel free to adapt the link to different versions or
locations.

Once `get_deps` has pulled each one of them, the next step would be to append the
include directories for these libraries to the `MLPACK_INCLUDE_DIRS` list
variable. In the case of Armadillo, it could be set in header-only mode or a
linkable library thus these two variable exist.

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

The last part of our CMake file consists of merging all of the above parts
together. Fist we are going to find the OpenMP package if the user defined the
variable above. and then we are going to start including the directories or our
program. In 

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
  /meta/mlpack/src/
)

## Do not forget to add the libraries that you are linking to in here.

target_link_libraries(RandomForest PRIVATE -static
  ${MLPACK_LIBRARIES}
)
```

If you are interested in adding specific compiler flags to optimize operations
on your hardware. Please feel free to look at mlpack CMake/flags you can either
integrate the variable by copy and paste them directly to this CMake file or you can
include ` ` as we have above but in this case you need to specify the
`BOARD_NAME` variable.

Once you have added all the source files and the headers for your applications,
you can create your own build directory and build the software using `cmake
../`, and then `make`.


