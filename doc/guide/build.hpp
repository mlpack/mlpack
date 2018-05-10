/*! @page build Building mlpack From Source

@section build_buildintro Introduction

This document discusses how to build mlpack from source.  However, mlpack is in
the repositories of many Linux distributions and so it may be easier to use the
package manager for your system.  For example, on Ubuntu, you can install mlpack
with the following command:

@code
$ sudo apt-get install libmlpack-dev
@endcode

If mlpack is not available in your system's package manager, then you can follow
this document for how to compile and install mlpack from source.

mlpack uses CMake as a build system and allows several flexible build
configuration options.  One can consult any of numerous CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed on most Linux and UNIX-like systems (including OS X).  If you want
to build mlpack on Windows, see <a
href="https://keon.io/mlpack-on-windows/">Keon's excellent tutorial</a>.

You can download the latest mlpack release from here:
<a href="http://www.mlpack.org/files/mlpack-3.0.1.tar.gz">mlpack-3.0.1</a>

@section build_simple Simple Linux build instructions

Assuming all dependencies are installed in the system, you can run the commands
below directly to build and install mlpack.

@code
$ wget http://www.mlpack.org/files/mlpack-3.0.1.tar.gz
$ tar -xvzpf mlpack-3.0.1.tar.gz
$ mkdir mlpack-3.0.1/build && cd mlpack-3.0.1/build
$ cmake ../
$ make -j4  # The -j is the number of cores you want to use for a build.
$ sudo make install
@endcode

If the \c cmake \c .. command fails, you are probably missing a dependency, so
check the output and install any necessary libraries.  (See \ref build_dep.)

The instructions above are the simplest way to get, build, and install mlpack.
The sections below discuss each of those steps in further detail and show how to
configure mlpack.

@section build_builddir Creating Build Directory

First we should unpack the mlpack source and create a build directory.

@code
$ tar -xvzpf mlpack-3.0.1.tar.gz
$ cd mlpack-3.0.1
$ mkdir build
@endcode

The directory can have any name, not just 'build', but 'build' is sufficient.

@section build_dep Dependencies of mlpack

mlpack depends on the following libraries, which need to be installed on the
system and have headers present:

 - Armadillo >= 6.500.0 (with LAPACK support)
 - Boost (math_c99, program_options, serialization, unit_test_framework, heap,
          spirit) >= 1.49

For Python bindings, the following packages are required:

 - setuptools
 - cython >= 0.24
 - numpy
 - pandas >= 0.15.0
 - pytest-runner

In Ubuntu and Debian, you can get all of these dependencies through apt:

@code
# apt-get install libboost-math-dev libboost-program-options-dev
  libboost-test-dev libboost-serialization-dev libarmadillo-dev binutils-dev
  python-pandas python-numpy cython python-setuptools
@endcode

On Fedora, Red Hat, or CentOS, these same dependencies can be obtained via dnf:

@code
# dnf install boost-devel boost-test boost-program-options boost-math
  armadillo-devel binutils-devel python2-Cython python2-setuptools
  python2-numpy python2-pandas
@endcode

(It's also possible to use python3 packages from the package manager---mlpack
will work with either.)

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
 - BUILD_SHARED_LIBRARIES=(ON/OFF): compile shared libraries as opposed to
       static libraries (default ON)
 - TEST_VERBOSE=(ON/OFF): run test cases in \c mlpack_test with verbose output
       (default OFF)
 - MATHJAX=(ON/OFF): use MathJax for generated Doxygen documentation (default
       OFF)
 - FORCE_CXX11=(ON/OFF): assume that the compiler supports C++11 instead of
       checking; be sure to specify any necessary flag to enable C++11 as part
       of CXXFLAGS (default OFF)

Each option can be specified to CMake with the '-D' flag.  Other tools can also
be used to configure CMake, but those are not documented here.

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

http://mlpack.org/

http://github.com/mlpack/mlpack

Alternately, mlpack help can be found in IRC at \#mlpack on irc.freenode.net.

@section install Installing mlpack

If you wish to install mlpack to the system, make sure you have root privileges
(or write permissions to those two directories), and simply type

@code
# make install
@endcode

You can now run the executables by name; you can link against mlpack with
\c -lmlpack, and the mlpack headers are found in \c /usr/include or
\c /usr/local/include (depending on the system and CMake configuration).

*/
