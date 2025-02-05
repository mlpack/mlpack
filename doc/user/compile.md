# Compile an mlpack program

Once an mlpack application has been developed, it is easy to compile it into a
standalone program.  On this page, compilation is performed via the command-line
on a standard Linux or OS X system; if this is not your environment, see also:

 * [Cross-compile to a Raspberry Pi 2](../embedded/crosscompile_armv7.md)
 * [Deploy mlpack on Windows](deploy_windows.md)

## Simple command-line compilation

Assuming that mlpack and its dependencies are [installed on the
system](install.md), an mlpack program can be compiled just like any other C++
program:

```sh
g++ -std=c++17 -O3 -o mlpack_program mlpack_program.cpp -larmadillo -fopenmp
```

The command above uses [gcc](https://gcc.gnu.org/) to compile the program
`mlpack_program.cpp` in C++17 mode with optimizations, using OpenMP for
parallelization.  It is expected that `mlpack_program.cpp` has the `int main()`
function defined.

For more complex applications that have multiple source files, it can often be
easier to develop a simple [`Makefile`](https://www.gnu.org/software/make/manual/html_node/Simple-Makefile.html).

The [examples repository](https://github.com/mlpack/examples) contains several
standalone C++ projects, each of which have `Makefile`s.  These can be adapted
for any project, and are especially useful if any extra include directories or
library directories need to be specified.  (This might be the case if, for
instance, mlpack or any dependencies are not installed to standard locations.)

 * [Example adaptable `Makefile`](https://github.com/mlpack/examples/blob/master/cpp/neural_networks/mnist_cnn/Makefile)

A full list of compiler options to configure the build is beyond the scope of
this simple documentation, but
[this simple list](https://gist.github.com/g-berthiaume/74f0485fbba5cc3249eee458c1d0d386)
has a handful of commonly-used gcc/clang options.

### Configuring mlpack with compile-time definitions

Several compilation options can control the behavior of an mlpack program.
These can be specified directly on the command line, or at the top of the
program (before including mlpack or Armadillo!).

| ***Command-line option*** | ***Code option*** | ***Meaning*** |
|---------------------------|-------------------|---------------|
|*Speed and debugging.* |||
| `-DNDEBUG` | `#define NDEBUG` | Remove all debugging checks.  This can result in slightly faster code, but with no error checking! |
| `-DARMA_NO_DEBUG` | `#define ARMA_NO_DEBUG` | Remove all Armadillo error checking.  *Warning:* if there are errors in your code, you are more likely to get a segfault instead of an exception! |
|---------------------------|-------------------|---------------|
|*Output.* |||
| `-DMLPACK_COUT_STREAM=std::cout` | `#define MLPACK_COUT_STREAM std::cout` | Set the default output stream.  (Defaults to `std::cout`.) |
| `-DMLPACK_CERR_STREAM=std::cerr` | `#define MLPACK_CERR_STREAM std::cerr` | Set the default error stream.  (Defaults to `std::cerr`.) |
| `-DMLPACK_PRINT_INFO` | `#define MLPACK_PRINT_INFO` | Print information messages (`[INFO ]`) during program execution. |
| `-DMLPACK_PRINT_WARN` | `#define MLPACK_PRINT_WARN` | Print warning messages (`[WARN ]`) during program execution. |
| `-DMLPACK_SUPPRESS_FATAL` | `#define MLPACK_PRINT_FATAL` | Do not print `[FATAL]` messages during program execution. |
| `-DENS_PRINT_INFO` | `#define ENS_PRINT_INFO` | Print informational messages from [ensmallen](https://www.ensmallen.org/) optimizers. |
| `-DENS_PRINT_WARN` | `#define ENS_PRINT_WARN` | Print warning messages from [ensmallen](https://ensmallen.org/) optimizers. |
|---------------------------|-------------------|---------------|
|*Functionality.* |||
| `-DMLPACK_ENABLE_ANN_SERIALIZATION` | `#define MLPACK_ENABLE_ANN_SERIALIZATION` | Allow neural network layers to be serialized. |
| `-DMLPACK_NO_STD_MUTEX` | `#define MLPACK_NO_STD_MUTEX` | Disable mutexes inside mlpack; use this if your system has no support for `std::mutex` and has only one core.  You may also need to define `ARMA_DO_NOT_USE_STD_MUTEX` for Armadillo. |
|---------------------------|-------------------|---------------|
|*Configuration.* |||
| `-DMLPACK_USE_SYSTEM_STB` | `#define MLPACK_USE_SYSTEM_STB` | Use the version of STB available on the system instead of the version bundled with mlpack.  If set, make sure `stb_image.h`, `stb_image_write.h`, and `stb_image_resize2.h` are available. |
| `-DMLPACK_DONT_USE_SYSTEM_STB` | `#define MLPACK_DONT_USE_SYSTEM_STB` | Force usage of the bundled version of STB.  Only necessary if mlpack was [configured](install.md#cmake-options) with `USE_SYSTEM_STB=ON`. |

***Note:*** If your code serializes (saves or loads) mlpack neural networks, the
`MLPACK_ENABLE_ANN_SERIALIZATION` option must be enabled.  This option is not
enabled by default because it can cause compilation time to increase
significantly, but it is necessary for any code that serializes neural networks.

## Linking without the Armadillo wrapper

Armadillo, by default, requires linking against the runtime library
`libarmadillo.so` (or `libarmadillo.dylib` or `armadillo.dll` on non-Linux
systems).  This library is a convenience library that internally contains all of
the symbols necessary from lower-level libraries (e.g.
[OpenBLAS](https://www.openblas.net/),
[SuperLU](https://portal.nersc.gov/project/sparse/superlu/),
[ARPACK](https://www.arpack.org/),
[HDF5](https://www.hdfgroup.org/solutions/hdf5/), and so on).
When the wrapper library is used, linking against Armadillo means simply typing
`-larmadillo` instead of linking against all of Armadillo's dependencies.

In some situations this is not preferable, and it is therefore possible via the
[`ARMA_DONT_USE_WRAPPER` macro](https://arma.sourceforge.net/docs.html#config_hpp)
to avoid the Armadillo runtime library and link directly against Armadillo's
dependencies.

When the Armadillo wrapper library is not being used, a compilation command will
need to be adjusted.  For instance, the example of the previous section would
need to be changed to:

```
g++ -DARMA_DONT_USE_WRAPPER -std=c++17 -O3 -o mlpack_program mlpack_program.cpp -lopenblas -fopenmp
```

Some notes on the command above:

 * Here, `ARMA_DONT_USE_WRAPPER` is specified on the command line instead of in
   `mlpack_program.cpp` (or otherwise in the Armadillo
    [configuration](https://arma.sourceforge.net/docs.html#config_hpp)).

 * OpenBLAS is used for BLAS/LAPACK support.  But, other options include ACML,
   reference LAPACK/BLAS, Intel MKL, and so forth.

 * In some programs, especially if sparse matrix support or HDF5 support is
   used, it may be necessary to link against other libraries (e.g. `-lSuperLU
   -lhdf5`, etc.).  The precise set of libraries to link against depends on the
   code being used and the system configuration, but it should be easy enough to
   use any linker errors to figure out what libraries need to be linked against.

## Using mlpack in another CMake project

For complex C++ projects, a build system like CMake may be in use.  Adding
mlpack as a dependency to a C++ project is straightforward.  The following CMake
code will require mlpack and its dependencies to be available:

```cmake
# Find mlpack and its dependencies.
find_package(Armadillo REQUIRED)
find_package(cereal REQUIRED)
find_package(ensmallen REQUIRED)
find_package(mlpack REQUIRED)

include_directories("${ARMADILLO_INCLUDE_DIRS}" "${CEREAL_INCLUDE_DIR}"
    "${ENSMALLEN_INCLUDE_DIR}" "${MLPACK_INCLUDE_DIR}")

# Targets should link against ${ARMADILLO_LIBRARIES}.
```

If the relevant files are not available on the system to find those four
packages, they can be downloaded from the
[models](https://github.com/mlpack/models) repository:

 * [`models/CMake` directory](https://github.com/mlpack/models/tree/master/CMake)

The following files in that directory are necessary (and can be added to the
CMake files for the project):

 * [`ARMA_FindACML.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindACML.cmake)
 * [`ARMA_FindACMLMP.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindACMLMP.cmake)
 * [`ARMA_FindARPACK.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindARPACK.cmake)
 * [`ARMA_FindBLAS.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindBLAS.cmake)
 * [`ARMA_FindCBLAS.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindCBLAS.cmake)
 * [`ARMA_FindCLAPACK.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindCLAPACK.cmake)
 * [`ARMA_FindLAPACK.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindLAPACK.cmake)
 * [`ARMA_FindMKL.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindMKL.cmake)
 * [`ARMA_FindOpenBLAS.cmake`](https://github.com/mlpack/models/blob/master/CMake/ARMA_FindOpenBLAS.cmake)
 * [`FindArmadillo.cmake`](https://github.com/mlpack/models/blob/master/CMake/FindArmadillo.cmake)
 * [`FindEnsmallen.cmake`](https://github.com/mlpack/models/blob/master/CMake/FindEnsmallen.cmake)
 * [`Findcereal.cmake`](https://github.com/mlpack/models/blob/master/CMake/Findcereal.cmake)
 * [`Findmlpack.cmake`](https://github.com/mlpack/models/blob/master/CMake/Findmlpack.cmake)

<!-- TODO: improve this so that it is simpler in the future! -->
