# Installing mlpack

mlpack is available via a wide variety of sources, depending on what you want to
do with the library.

***If you want to use mlpack in a C++ program:***

 * Install via [your system's package manager](#install-via-package-manager)
   *(easiest)*.

 * [Install from source](#install-from-source) (see also the
   [dependencies](#dependencies) of mlpack).

 * If you are on Windows, see the
   [Building mlpack from source on Windows page](build_windows.md).

 * If you intend to cross-compile, see the
   [cross-compilation setup page](../embedded/supported_boards.md).

***If you want to use mlpack's bindings to another language:***

 * Install mlpack's bindings to other languages via
   [language package managers](#install-bindings-via-language-package-managers)
   *(easiest)*.

 * [Compile and install bindings manually](#compile-bindings-manually).

***If you want to develop mlpack:***

 * [Configure and compile all of mlpack from source](#compile-from-source).

 * Look at the [CMake configuration options](#cmake-options).

 * [Build the tests](#build-tests).

Once mlpack is installed, try
[compiling a test program](#compiling-a-test-program).

---

## Install via package manager

The easiest way to install the mlpack C++ library is to use your system package
manager.  This will handle mlpack's dependencies automatically.

 * ***Ubuntu/Debian***: `sudo apt-get install libmlpack-dev`
 * ***Fedora/RHEL***: `sudo dnf install mlpack-devel`
 * ***Arch Linux***: `sudo pacman -S mlpack`
 * ***OS X (Homebrew)***: `brew install mlpack`
 * ***OS X (MacPorts)***: `sudo port install mlpack`
 * ***vcpkg (Windows)***: `vcpkg install mlpack:x64-windows`
 * ***conda***: `conda install conda-forge::mlpack`
 * ***Conan***: [see here](https://conan.io/center/recipes/mlpack)

You can also use the
[`mlpack/mlpack` image on DockerHub](https://hub.docker.com/r/mlpack/mlpack) for
a container with mlpack already installed.

If you plan to write mlpack programs, make sure you have a C++ compiler that
supports C++17 available (this may not be automatically installed by the package
manager).

***Warning:*** on Ubuntu and Debian systems, older versions of OpenBLAS (0.3.26
and older) can over-use the number of cores on your system, causing slow
execution of mlpack programs, especially mlpack's test suite.  To prevent this,
set `OMP_NUM_THREADS` as detailed [in the test build
guide](../user/install.md#build-tests), or install the `libopenblas-openmp-dev`
package on Ubuntu or Debian and remove `libopenblas-pthread-dev`.  Ubuntu 24.04,
Debian bookworm, and older are all affected by this issue.

## Install from source

If you only intend to use mlpack in a C++ program, it is not necessary to
[configure and compile from source](#compile-from-source), because mlpack is a
header-only library.  This means that you can simply
[download mlpack](https://www.mlpack.org/download.html) and unpack it, and when
you are [compiling a program](#compiling-a-test-program), you must make sure
that the `src/` directory is on the include path.

With most compilers, this means you simply add the flag `-I/path/to/mlpack/src/`
to the compiler command-line (e.g.,
`g++ -I/path/to/mlpack/src/ -o program program.cpp -larmadillo`).

---

If you wish to install the mlpack headers to your system manually via CMake, you
can use the following commands:

```sh
mkdir build && cd build/
cmake ..
sudo make install
```

Alternately, since CMake v3.14.0, the `cmake` command can create the build
folder itself, and so the above commands can be rewritten as follows:

```sh
cmake -S . -B build
sudo cmake --build build --target install
```

### Dependencies

You must also ensure that the dependencies of mlpack are available to the
compiler:

 - [Armadillo](https://arma.sourceforge.net)      &nbsp;&emsp;>= 10.8
 - [ensmallen](https://ensmallen.org)      &emsp;>= 2.10.0
 - [cereal](http://uscilab.github.io/cereal/)         &ensp;&nbsp;&emsp;&emsp;>= 1.1.2

Dependencies can be installed using the system package manager.   For example,
on Debian and Ubuntu, all relevant dependencies can be installed with `sudo
apt-get install libarmadillo-dev libensmallen-dev libcereal-dev libstb-dev g++
cmake`.

If the STB library headers are available, image loading support will be
available.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

If you are configuring mlpack with CMake (as in the code snippets in the
previous section), you can use the auto-downloader to obtain mlpack's
dependencies with the `-DDOWNLOAD_DEPENDENCIES=ON` option (detailed in the
[CMake options section](#cmake-options).  The autodownloader is especially
useful for [cross-compilation](../embedded/supported_boards.md), as it
automatically downloads and compiles OpenBLAS for the target architecture.

## Install bindings via language package managers

If you wish to use mlpack's bindings to other languages, see the quickstarts for
each language for more information on installation:

 * [Python](../quickstart/python.md)
 * [Command-line](../quickstart/cli.md)
 * [Julia](../quickstart/julia.md)
 * [R](../quickstart/r.md)
 * [Go](../quickstart/go.md)

## Compile bindings manually

It is possible to manually build the bindings from source.  However, this is not
recommended, as building bindings for a specific language often requires some
amount of setup and is not often a user-friendly process.  Specifically, after
the bindings are built, deploying them to the environment of the target language
can be non-trivial and requires knowledge specific to that language (not covered
here).

To compile bindings for a particular language, follow the
[Compile from source](#compile-from-source) section below, and enable the
appropriate [CMake options](#cmake-options).

The results of bindings will be built into `build/src/mlpack/bindings/<lang>/`
where `build/` is the build directory configured with CMake, and `<lang>` should
be replaced with the appropriate language (`python`/`r`/`go`/`julia`).  There
are two exceptions:

 * Command-line bindings will be built into `build/bin/`.
 * Markdown bindings will produce Markdown files in `build/doc/`.

## Compile from source

If you intend to develop mlpack, or want to build the tests or bindings to
another language (including the command-line bindings), you will need to compile
from source.  Once you have installed [the dependencies](#dependencies) and
[downloaded mlpack](https://www.mlpack.org/download.html), unpack the sources
and configure with CMake.

The command below enables building the tests and the command-line programs.
More options are detailed in the [CMake options section](#cmake-options).

```sh
mkdir build && cd build/
cmake -DBUILD_TESTS=ON -DBUILD_CLI_EXECUTABLES=ON ../
make -j4
```

The `-j4` option specifies that 4 cores should be used for the build; if you are
running into RAM limitations (or don't have four cores), reduce this.  If you
have more cores available, you can increase the number of cores for a faster
build.

### CMake options

The following options can be used when configuring mlpack.

| ***Option*** | ***Description*** | ***Default*** |
|--------------|-------------------|---------------|
| ***General configuration*** |||
| `-DCMAKE_BUILD_TYPE="build type"` | Specify the build configuration: `"Debug"`, `"Release"`, `"RelWithDebInfo"`, `"MinSizeRel"`.  See the [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html#variable:CMAKE_BUILD_TYPE). | `"Release"`|
| `-DDOWNLOAD_DEPENDENCIES=ON` | Download all dependencies that are not found on the system. | `OFF` |
| `-DARMA_EXTRA_DEBUG=ON` | Emit extra Armadillo debugging output (warning: *very* verbose). | `OFF` |
| `-DTEST_VERBOSE=ON` | Emit verbose output when running tests. | `OFF` |
| `-DBUILD_TESTS=ON` | Build `mlpack_test`. | `OFF` |
| `-DUSE_OPENMP=ON` | Use OpenMP for parallelization. | `ON` |
| `-DUSE_PRECOMPILED_HEADERS=OFF` | Disable precompiled headers during build. | `OFF` |
| `-DUSE_SYSTEM_STB=OFF` | Use version of STB bundled with mlpack. If set to `ON` make sure `stb_image.h`, `stb_image_write.h`, and `stb_image_resize2.h` are available. | `OFF` |
|--------------|-------------------|---------------|
| ***Dependency locations*** |||
| `-DARMADILLO_INCLUDE_DIR=/path/to/arma/include/` | Path containing `armadillo` header file. ||
| `-DARMADILLO_LIBRARY=/path/to/libarmadillo.so` | Path of compiled Armadillo library (if using the Armadillo wrapper library). ||
| `-DARMADILLO_LIBRARIES=/path/to/lib1.so;/path/to/lib2.so` | List of libraries to link against for Armadillo (if not using the Armadillo wrapper library). ||
| `-DCEREAL_INCLUDE_DIR=/path/to/cereal/include/` | Path containing cereal headers. ||
| `-DENSMALLEN_INCLUDE_DIR=/path/to/ens/include/` | Path containing `ensmallen.hpp`. ||
| `-DSTB_INCLUDE_DIR=/path/to/stb/include/` | Path containing `stb.h` and `stb_image.h`. ||
|--------------|-------------------|---------------|
| ***Bindings*** |||
| `-DBUILD_CLI_EXECUTABLES=ON` | Enable building command-line programs. | `OFF` |
| `-DBUILD_PYTHON_BINDINGS=ON` | Enable building Python bindings. | `OFF` |
| `-DPYTHON_EXECUTABLE=/path/to/python` | Location of Python program to use. ||
| `-DBUILD_GO_BINDINGS=ON` | Enable building Go bindings. | `OFF` |
| `-DBUILD_GO_SHLIB=OFF` | Do not shared library for Go bindings. | `ON` |
| `-DBUILD_JULIA_BINDINGS=ON` | Enable building Julia bindings. | `OFF` |
| `-DJULIA_EXECUTABLE=/path/to/julia` | Location of Julia interpreter. ||
| `-DBUILD_R_BINDINGS=ON` | Enable building R bindings. | `OFF` |
| `-DBUILD_MARKDOWN_BINDINGS=ON` | Enable building Markdown bindings (e.g. Markdown documentation for each binding language). | `OFF` |
|--------------|-------------------|---------------|

### Build tests

If you are developing mlpack or simply want to run the test suite, after you
have configured the library with CMake, you can build the tests directly:

```sh
make -j4 mlpack_test
```

Replace the `-j4` with the number of cores desired for building.

Once the build is complete (it may take a while!), you can run the tests from
the build directory, selecting either all of them or an individual test suite.

```sh
bin/mlpack_test
bin/mlpack_test [LARSTest]
```

The `mlpack_test` program uses the [Catch2](https://github.com/catchorg/Catch2)
library for unit testing; this supports many options---you can see them with
`mlpack_test -h`.

***Warning:*** on Linux systems, older versions of OpenBLAS (0.3.26 and older)
compiled to use pthreads can over-use the number of cores on your system,
causing slow execution of mlpack programs, especially mlpack's test suite.
OpenBLAS versions compiled with OpenMP do not suffer from this issue.  To work
around this problem, set `OMP_NUM_THREADS` to half the number of cores on your
system; so, for a system with 8 cores, run

```sh
OMP_NUM_THREADS=4 bin/mlpack_test
```

This workaround should be applied to any mlpack program where:

 * Compilation with OpenMP is enabled (e.g. `-fopenmp` is used),
 * Armadillo is using OpenBLAS,
 * OpenBLAS is version 0.3.26 or older, and
 * OpenBLAS is compiled against pthreads (the default on Ubuntu, Debian, and
   Fedora).

## Compiling a test program

Once mlpack is installed and available on the system, it is easy to compile a
program using mlpack.  For instance, consider the trivial program below:

```c++
#include <mlpack.hpp>

using namespace mlpack;

int main()
{
  // Sample a point from a 3-dimensional Gaussian distribution.
  GaussianDistribution g(3);
  std::cout << "Random sample from 3D Gaussian: " << std::endl
      << g.Random();
}
```

This can be compiled with the command:

```
g++ -O3 -std=c++17 -o my_program my_program.cpp -larmadillo -fopenmp
```

The command may need slight adaptation if you are using a different compiler or
prefer different compilation options.

***Notes***:

 - If you want to serialize (save or load) neural networks, you should add
   `#define MLPACK_ENABLE_ANN_SERIALIZATION` before including `<mlpack.hpp>`.

 - When the autodownloader is used to download Armadillo
   (`-DDOWNLOAD_DEPENDENCIES=ON`), the Armadillo runtime library is not built
   and Armadillo must be used in header-only mode. Instead, you must link
   directly with the dependencies of Armadillo.  For example, on a system that
   has OpenBLAS available, compilation can be done like this:

```sh
g++ -O3 -std=c++17 -o my_program my_program.cpp -lopenblas -fopenmp
```

 - See also the warning on OpenBLAS versions and too many threads in [the
   previous section](#build-tests).

See [the Armadillo documentation](https://arma.sourceforge.net/faq.html#linking)
for more information on linking Armadillo programs.
