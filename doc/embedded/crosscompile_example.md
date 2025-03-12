## Crosscompile mlpack example for an embedded hardware

In this article, we explore how to add mlpack to a CMake project that cross-compiles
code to embedded hardware.  See also these related other guides, which may be useful
to read before this one:

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

### `CMakeLists.txt`

The first part of the code, printed below or
[available here](https://github.com/mlpack/examples/blob/master/embedded/crosscompile_random_forest/CMakeLists.txt),
defines the project name and includes two useful CMake configuration files:
 * `mlpack.cmake`: finds mlpack's dependencies and download them if necessary.
 * `ConfigureCrossCompile.cmake`: set up CMake configuration for cross-compilation.
 * `crosscompile-toolchain.cmake`: invoke CMake crosscompilation infrastructure.
 * `crosscompile-arch-config.cmake`: add necessary flags depending on the
   architecture (optional).

Then we need to call `fetch_mlpack(ON)` to download mlpack including all dependencies,
cross-compile OpenBLAS and set up all the necessary parameters to find these dependencies.
Most of mlpack's dependencies are header-only with the exception of OpenBLAS;
thus this is expected to be a quick step.

`fetch_mlpack()` will detect if cross compilation is necessary or not depending
on the command that is executed when running cmake. Based on this, it will
compile OpenBLAS.

```cmake
cmake_minimum_required(VERSION 3.11)
project(main)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
include(CMake/mlpack.cmake)
include(CMake/ConfigureCrossCompile.cmake)

// Download all of mlpack's dependencies and cross-compile OpenBLAS.
fetch_mlpack(ON)
```

#### Setting up include directories and source files

The last part of our `CMakeFiles.txt` consists of merging all of the components above
together. First we need to start including the directories of our
program. You will need to do the same for your software if you are trying to
integrate this example into an existing codebase; this should be done by
defining an include variable using `set` directive or add them directly in
`target_include_directories`.
Regarding the source code, a similar process by either adding then to
`add_executables` or by appending the source files to `SOURCES_FILES` variable. 

Finally do not forget to add any external library that you need to link against
in `target_link_libraries`.

```cmake
## Add your source files to SOURCES_FILES list
set(SOURCE_FILES main.cpp)
add_executable(RandomForest main.cpp ${SOURCES_FILES})

# If needed, add any additional include directories here.
target_include_directories(RandomForest PRIVATE
  ${MLPACK_INCLUDE_DIRS}
  # Add more include directories here...
)

# If your application needs to link against more than just mlpack's
# dependencies, be sure to list them here.
target_link_libraries(RandomForest PRIVATE -static
  ${MLPACK_LIBRARIES}
  # List additional dependencies to link against here.
)
```

#### Optimization and cross-compilation

If you are interested in adding specific compiler flags to optimize operations
on your hardware, you can either set `CMAKE_CXX_FLAGS` manually, or look at the
copy of `CMake/crosscompile-arch-flags.cmake` in your project, find the
appropriate `ARCH_NAME` section, and add the new compilation flags to that file.

Once you have added all the source files and the headers for your applications,
you can create your own build directory and build the software using `cmake`,
and then `make`. Your cmake command should be similar to the following:

```sh
cmake \
    -DARCH_NAME=(Check below) \
    -DCMAKE_CROSSCOMPILING=ON \
    -DCMAKE_TOOLCHAIN_FILE=../CMake/crosscompile-toolchain.cmake \
    -DTOOLCHAIN_PREFIX=(Check below) \
    -DCMAKE_SYSROOT=(Check below) \
```

In order to fill the `TOOLCHAIN_PREFIX` and `CMAKE_SYSROOT` variables, use
[this table](supported_boards.md).

If your preferred architecture is missing, or if the table needs an update,
please submit a PR to the repository and help us keep it up to date!

