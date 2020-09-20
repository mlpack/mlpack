/*! @page build Building mlpack From Source

@section build_buildintro Introduction

This document discusses how to build mlpack from source. These build directions 
will work for any Linux-like shell environment (for example Ubuntu, macOS,
FreeBSD etc). However, mlpack is in the repositories of many Linux distributions 
and so it may be easier to use the package manager for your system.  For example, 
on Ubuntu, you can install mlpack with the following command:

@code
$ sudo apt-get install libmlpack-dev
@endcode

@note Older Ubuntu versions may not have the most recent version of mlpack
available---for instance, at the time of this writing, Ubuntu 16.04 only has
mlpack 2.0.1 available.  Options include upgrading Ubuntu to a newer release,
finding a PPA or other non-official sources, or installing with a manual build
(below).

If mlpack is not available in your system's package manager, then you can follow
this document for how to compile and install mlpack from source.

mlpack uses CMake as a build system and allows several flexible build
configuration options.  One can consult any of numerous CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed on most Linux and UNIX-like systems (including OS X).  If you want
to build mlpack on Windows, see \ref build_windows (alternatively, you can read 
<a href="https://keon.io/mlpack-on-windows/">Keon's excellent tutorial</a> which
is based on older versions).

You can download the latest mlpack release from here:
<a href="https://www.mlpack.org/files/mlpack-3.4.1.tar.gz">mlpack-3.4.1</a>

@section build_simple Simple Linux build instructions

Assuming all dependencies are installed in the system, you can run the commands
below directly to build and install mlpack.

@code
$ wget https://www.mlpack.org/files/mlpack-3.4.1.tar.gz
$ tar -xvzpf mlpack-3.4.1.tar.gz
$ mkdir mlpack-3.4.1/build && cd mlpack-3.4.1/build
$ cmake ../
$ make -j4  # The -j is the number of cores you want to use for a build.
$ sudo make install
@endcode

If the \c cmake \c .. command fails, you are probably missing a dependency, so
check the output and install any necessary libraries.  (See \ref build_dep.)

On many Linux systems, mlpack will install by default to @c /usr/local/lib and
you may need to set the @c LD_LIBRARY_PATH environment variable:

@code
export LD_LIBRARY_PATH=/usr/local/lib
@endcode

The instructions above are the simplest way to get, build, and install mlpack.
The sections below discuss each of those steps in further detail and show how to
configure mlpack.

@section build_builddir Creating Build Directory

First we should unpack the mlpack source and create a build directory.

@code
$ tar -xvzpf mlpack-3.4.1.tar.gz
$ cd mlpack-3.4.1
$ mkdir build
@endcode

The directory can have any name, not just 'build', but 'build' is sufficient.

@section build_dep Dependencies of mlpack

mlpack depends on the following libraries, which need to be installed on the
system and have headers present:

 - Armadillo >= 8.400.0 (with LAPACK support)
 - Boost (math_c99, serialization, unit_test_framework, heap,
          spirit) >= 1.58
 - ensmallen >= 2.10.0 (will be downloaded if not found)

In addition, mlpack has the following optional dependencies:

 - STB: this will allow loading of images; the library is downloaded if not
   found and the CMake variable DOWNLOAD_STB_IMAGE is set to ON (the default)

For Python bindings, the following packages are required:

 - setuptools
 - cython >= 0.24
 - numpy
 - pandas >= 0.15.0
 - pytest-runner

In Ubuntu (>= 18.04) and Debian (>= 10) all of these dependencies can be 
installed through apt:

@code
# apt-get install libboost-math-dev libboost-test-dev libboost-serialization-dev
  libarmadillo-dev binutils-dev python3-pandas python3-numpy cython3
  python3-setuptools
@endcode

If you are using Ubuntu 19.10 or newer, you can also install @c libensmallen-dev
and @c libstb-dev, so that CMake does not need to automatically download those
packages:

@code
# apt-get install libensmallen-dev libstb-dev
@endcode

@note For older versions of Ubuntu and Debian, Armadillo needs to be built from 
source as apt installs an older version. So you need to omit 
\c libarmadillo-dev from the code snippet above and instead use
<a href="http://arma.sourceforge.net/download.html">this link</a>
 to download the required file. Extract this file and follow the README in the 
 uncompressed folder to build and install Armadillo.

On Fedora, Red Hat, or CentOS, these same dependencies can be obtained via dnf:

@code
# dnf install boost-devel boost-test boost-math armadillo-devel binutils-devel 
  python3-Cython python3-setuptools python3-numpy python3-pandas ensmallen-devel 
  stbi-devel
@endcode

(It's also possible to use python3 packages from the package manager---mlpack
will work with either.  Also, the ensmallen-devel package is only available in
Fedora 29 or RHEL7 or newer.)

@section build_config Configuring CMake

Running CMake is the equivalent to running `./configure` with autotools.  If you
run CMake with no options, it will configure the project to build without
debugging or profiling information (for speed).

@code
$ cd build
$ cmake ../
@endcode

You can manually specify options to compile with debugging information and
profiling information (useful if you are developing mlpack):

@code
$ cd build
$ cmake -D DEBUG=ON -D PROFILE=ON ../
@endcode

The full list of options mlpack allows:

 - DEBUG=(ON/OFF): compile with debugging symbols (default OFF)
 - PROFILE=(ON/OFF): compile with profiling symbols (default OFF)
 - ARMA_EXTRA_DEBUG=(ON/OFF): compile with extra Armadillo debugging symbols
       (default OFF)
 - BUILD_TESTS=(ON/OFF): compile the \c mlpack_test program (default ON)
 - BUILD_CLI_EXECUTABLES=(ON/OFF): compile the mlpack command-line executables
       (i.e. \c mlpack_knn, \c mlpack_kfn, \c mlpack_logistic_regression, etc.)
       (default ON)
 - BUILD_PYTHON_BINDINGS=(ON/OFF): compile the bindings for Python, if the
       necessary Python libraries are available (default ON except on Windows)
 - BUILD_JULIA_BINDINGS=(ON/OFF): compile Julia bindings, if Julia is found
       (default ON)
 - BUILD_SHARED_LIBS=(ON/OFF): compile shared libraries as opposed to
       static libraries (default ON)
 - TEST_VERBOSE=(ON/OFF): run test cases in \c mlpack_test with verbose output
       (default OFF)
 - DISABLE_DOWNLOADS=(ON/OFF): Disable downloads of dependencies during build
       (default OFF)
 - DOWNLOAD_ENSMALLEN=(ON/OFF): If ensmallen is not found, download it
       (default ON)
 - DOWNLOAD_STB_IMAGE=(ON/OFF): If STB is not found, download it (default ON)
 - BUILD_WITH_COVERAGE=(ON/OFF): Build with support for code coverage tools
      (gcc only) (default OFF)
 - PYTHON_EXECUTABLE=(/path/to/python_version): Path to specific Python executable
 - JULIA_EXECUTABLE=(/path/to/julia): Path to specific Julia executable
 - BUILD_MARKDOWN_BINDINGS=(ON/OFF): Build Markdown bindings for website
       documentation (default OFF)
 - MATHJAX=(ON/OFF): use MathJax for generated Doxygen documentation (default
       OFF)
 - FORCE_CXX11=(ON/OFF): assume that the compiler supports C++11 instead of
       checking; be sure to specify any necessary flag to enable C++11 as part
       of CXXFLAGS (default OFF)
 - USE_OPENMP=(ON/OFF): if ON, then use OpenMP if the compiler supports it; if
       OFF, OpenMP support is manually disabled (default ON)

Each option can be specified to CMake with the '-D' flag.  Other tools can also
be used to configure CMake, but those are not documented here.

In addition, the following directories may be specified, to find include files
and libraries. These also use the '-D' flag.

 - ARMADILLO_INCLUDE_DIR=(/path/to/armadillo/include/): path to Armadillo headers
 - ARMADILLO_LIBRARY=(/path/to/armadillo/libarmadillo.so): location of Armadillo
       library
 - BOOST_ROOT=(/path/to/boost/): path to root of boost installation
 - ENSMALLEN_INCLUDE_DIR=(/path/to/ensmallen/include): path to include directory
       for ensmallen
 - STB_IMAGE_INCLUDE_DIR=(/path/to/stb/include): path to include directory for
      STB image library
 - MATHJAX_ROOT=(/path/to/mathjax): path to root of MathJax installation

@section build_build Building mlpack

Once CMake is configured, building the library is as simple as typing 'make'.
This will build all library components as well as 'mlpack_test'.

@code
$ make
Scanning dependencies of target mlpack
[  1%] Building CXX object
src/mlpack/CMakeFiles/mlpack.dir/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.cpp.o
<...>
@endcode

It's often useful to specify \c -jN to the \c make command, which will build on
\c N processor cores.  That can accelerate the build significantly.

You can specify individual components which you want to build, if you do not
want to build everything in the library:

@code
$ make mlpack_pca mlpack_knn mlpack_kfn
@endcode

One particular component of interest is mlpack_test, which runs the mlpack test
suite.  You can build this component with

@code
$ make mlpack_test
@endcode

and then run all of the tests, or an individual test suite:

@code
$ bin/mlpack_test
$ bin/mlpack_test -t KNNTest
@endcode

If the build fails and you cannot figure out why, register an account on Github
and submit an issue and the mlpack developers will quickly help you figure it
out:

https://mlpack.org/

https://github.com/mlpack/mlpack

Alternately, mlpack help can be found in IRC at \#mlpack on chat.freenode.net.

@section install Installing mlpack

If you wish to install mlpack to the system, make sure you have root privileges
(or write permissions to those two directories), and simply type

@code
# make install
@endcode

You can now run the executables by name; you can link against mlpack with
\c -lmlpack, and the mlpack headers are found in \c /usr/include or
\c /usr/local/include (depending on the system and CMake configuration).  If
Python bindings were installed, they should be available when you start Python.

@section build_run Using mlpack without installing

If you would prefer to use mlpack after building but without installing it to
the system, this is possible.  All of the command-line programs in the
@c build/bin/ directory will run directly with no modification.

For running the Python bindings from the build directory, the situation is a
little bit different.  You will need to set the following environment variables:

@code
export LD_LIBRARY_PATH=/path/to/mlpack/build/lib/:${LD_LIBRARY_PATH}
export PYTHONPATH=/path/to/mlpack/build/src/mlpack/bindings/python/:${PYTHONPATH}
@endcode

(Be sure to substitute the correct path to your build directory for
`/path/to/mlpack/build/`.)

Once those environment variables are set, you should be able to start a Python
interpreter and `import mlpack`, then use the Python bindings.

*/
