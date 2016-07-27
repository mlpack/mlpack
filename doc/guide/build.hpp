/*! @page build Building mlpack From Source

@section buildintro Introduction

mlpack uses CMake as a build system and allows several flexible build
configuration options.  One can consult any of numerous CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed on most Linux and UNIX-like systems (including OS X).  If you want
to build mlpack on Windows, see <a
href="http://keon.io/mlpack-on-windows.html">Keon's excellent tutorial</a>.

@section Download latest mlpack build
Download latest mlpack build from here:
<a href="http://www.mlpack.org/files/mlpack-2.0.2.tar.gz">mlpack-2.0.2</a>

@section builddir Creating Build Directory

Once the mlpack source is unpacked, you should create a build directory.

@code
$ cd mlpack-2.0.2
$ mkdir build
@endcode

The directory can have any name, not just 'build', but 'build' is sufficient
enough.

@section dep Dependencies of mlpack

mlpack depends on the following libraries, which need to be installed on the
system and have headers present:

 - Armadillo >= 4.100.0 (with LAPACK support)
 - Boost (math_c99, program_options, serialization, unit_test_framework, heap)
      >= 1.49

In Ubuntu and Debian, you can get all of these dependencies through apt:

@code
# apt-get install libboost-math-dev libboost-program-options-dev
  libboost-test-dev libboost-serialization-dev libarmadillo-dev binutils-dev
@endcode

On Fedora, Red Hat, or CentOS, these same dependencies can be obtained via dnf:

@code
# dnf install boost-devel boost-test boost-program-options boost-math
  armadillo-devel binutils-devel
@endcode

@section config Configuring CMake

Running CMake is the equivalent to running `./configure` with autotools.  If you
are working with the svn trunk version of mlpack and run CMake with no options,
it will configure the project to build with debugging symbols and profiling
information:  If you are working with a release of mlpack, running CMake with no
options will configure the project to build without debugging or profiling
information (for speed).

@code
$ cd build
$ cmake ../
@endcode

You can manually specify options to compile with or without debugging
information and profiling information (i.e. as fast as possible):

@code
$ cd build
$ cmake -D DEBUG=OFF -D PROFILE=OFF ../
@endcode

The full list of options mlpack allows:

 - DEBUG=(ON/OFF): compile with debugging symbols (default ON in svn trunk, OFF
   in releases)
 - PROFILE=(ON/OFF): compile with profiling symbols (default ON in svn trunk,
   OFF in releases)
 - ARMA_EXTRA_DEBUG=(ON/OFF): compile with extra Armadillo debugging symbols
       (default OFF)
 - BUILD_TESTS=(ON/OFF): compile the \c mlpack_test program (default ON)
 - BUILD_CLI_EXECUTABLES=(ON/OFF): compile the mlpack command-line executables
       (i.e. \c mlpack_knn, \c mlpack_kfn, \c mlpack_logistic_regression, etc.)
       (default ON)
 - TEST_VERBOSE=(ON/OFF): run test cases in \c mlpack_test with verbose output
       (default OFF)

Each option can be specified to CMake with the '-D' flag.  Other tools can also
be used to configure CMake, but those are not documented here.

@section build Building mlpack

Once CMake is configured, building the library is as simple as typing 'make'.
This will build all library components as well as 'mlpack_test'.

@code
$ make
Scanning dependencies of target mlpack
[  1%] Building CXX object
src/mlpack/CMakeFiles/mlpack.dir/core/optimizers/aug_lagrangian/aug_lagrangian_test_functions.cpp.o
<...>
@endcode

You can specify individual components which you want to build, if you do not
want to build everything in the library:

@code
$ make mlpack_pca mlpack_knn mlpack_kfn
@endcode

If the build fails and you cannot figure out why, register an account on Trac
and submit a ticket and the mlpack developers will quickly help you figure it
out:

http://mlpack.org/

Alternately, mlpack help can be found in IRC at \#mlpack on irc.freenode.net.

@section install Installing mlpack

If you wish to install mlpack to /usr/include/mlpack/ and /usr/lib/ and
/usr/bin/, once it has built, make sure you have root privileges (or write
permissions to those two directories), and simply type

@code
# make install
@endcode

You can now run the executables by name; you can link against mlpack with
-lmlpack, and the mlpack headers are found in /usr/include/mlpack/.

*/
