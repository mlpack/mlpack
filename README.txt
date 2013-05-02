================================================================================

           mlpack: open-source scalable c++ machine learning library

================================================================================

0. Contents

  1. Introduction
  2. Citation details
  3. Dependencies
  4. Building mlpack from source
  5. Running mlpack programs
  6. Further documentation
  7. Bug reporting

================================================================================

1. Introduction

mlpack is an intuitive, fast, scalable C++ machine learning library, meant to be
a machine learning analog to LAPACK. It aims to implement a wide array of
machine learning methods and function as a "swiss army knife" for machine
learning researchers.

The mlpack website can be found at http://mlpack.org and contains numerous
tutorials and extensive documentation.  This README serves as a guide for what
mlpack is, how to install it, how to run it, and where to find more
documentation.  The website should be consulted for further information:

  http://www.mlpack.org/

  http://www.mlpack.org/tutorial.html  <-- tutorials
  http://www.mlpack.org/trac/          <-- development site (Trac)
  http://www.mlpack.org/doxygen.php    <-- API documentation

================================================================================

2. Citation details

If you use mlpack in your research or software, please cite mlpack using the
citation below (given in BiBTeX format):

@INPROCEEDINGS{mlpack2011,
  author    = {Ryan R. Curtin and James R. Cline and Neil P. Slagle and Matthew
      L. Amidon and Alexander G. Gray},
  title     = {{MLPACK: A Scalable C++ Machine Learning Library}},
  booktitle = {{BigLearning: Algorithms, Systems, and Tools for Learning at
      Scale}},
  year      = 2011
}

Citations are beneficial for the growth and improvement of mlpack.

================================================================================

3. Dependencies

mlpack has the following dependencies:

  Armadillo     >= 2.4.4
  LibXml2       >= 2.6.0
  Boost (program_options, math_c99, unit_test_framework, random)
  CMake         >= 2.8.5

All of those should be available in your distribution's package manager.  If
not, you will have to compile each of them by hand.  See the documentation for
each of those packages for more information.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

================================================================================

4. Building mlpack from source
   (see also http://www.mlpack.org/doxygen.php?doc=build.html )

mlpack uses CMake as a build system and allows several flexible build
configuration options. One can consult any of numerous CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed.

First, unpack the mlpack source and change into the unpacked directory.  Here we
use mlpack-x.y.z where x.y.z is the version.

$ tar -xzf mlpack-x.y.z.tar.gz
$ cd mlpack-x.y.z

Then, make a build directory.  The directory can have any name, not just
'build', but 'build' is sufficient.

$ mkdir build
$ cd build

The next step is to run CMake to configure the project.  Running CMake is the
equivalent to running `./configure` with autotools. If you run CMake with no
options, it will configure the project to build with no debugging symbols and no
profiling information:

$ cmake ../

You can specify options to compile with debugging information and profiling
information:

$ cmake -D DEBUG=ON -D PROFILE=ON ../

Options are specified with the -D flag.  A list of options allowed:

  DEBUG=(ON/OFF): compile with debugging symbols
  PROFILE=(ON/OFF): compile with profiling symbols
  ARMA_EXTRA_DEBUG=(ON/OFF): compile with extra Armadillo debugging symbols
  BOOST_ROOT=(/path/to/boost/): path to root of boost installation
  ARMADILLO_INCLUDE_DIR=(/path/to/armadillo/include/): path to Armadillo headers
  ARMADILLO_LIBRARY=(/path/to/armadillo/libarmadillo.so): Armadillo library

Other tools can also be used to configure CMake, but those are not documented
here.

Once CMake is configured, building the library is as simple as typing 'make'.
This will build all library components as well as 'mlpack_test'.

$ make

You can specify individual components which you want to build, if you do not
want to build everything in the library:

$ make pca allknn allkfn

If the build fails and you cannot figure out why, register an account on Trac
and submit a ticket and the mlpack developers will quickly help you figure it
out:

http://mlpack.org/trac/

Alternately, mlpack help can be found in IRC at #mlpack on irc.freenode.net.

If you wish to install mlpack to /usr/include/mlpack/ and /usr/lib/ and
/usr/bin/, once it has built, make sure you have root privileges (or write
permissions to those two directories), and simply type

# make install

You can now run the executables by name; you can link against mlpack with
-lmlpack, and the mlpack headers are found in /usr/include/mlpack/.

================================================================================

5. Running mlpack programs

After building mlpack, the executables will reside in build/bin/.  You can call
them from there, or you can install the library and (depending on system
settings) they should be added to your PATH and you can call them directly.  The
documentation below assumes the executables are in your PATH.

We consider the 'allknn' program, which finds the k nearest neighbors in a
reference dataset of all the points in a query set.  That is, we have a query
and a reference dataset. For each point in the query dataset, we wish to know
the k points in the reference dataset which are closest to the given query
point.

Alternately, if the query and reference datasets are the same, the problem can
be stated more simply: for each point in the dataset, we wish to know the k
nearest points to that point.

Each mlpack program has extensive help documentation which details what the
method does, what each of the parameters are, and how to use them:

$ allknn --help

Running allknn on one dataset (that is, the query and reference datasets are the
same) and finding the 5 nearest neighbors is very simple:

$ allknn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 5 -v

The -v (--verbose) flag is optional; it gives informational output.  It is not
unique to allknn but is available in all mlpack programs.  Verbose output also
gives timing output at the end of the program, which can be very useful.

================================================================================

6. Further documentation

The documentation given here is only a fraction of the available documentation
for mlpack.  If doxygen is installed, you can type 'make doc' to build the
documentation locally.  Alternately, up-to-date documentation is available for
older versions of mlpack:

  http://www.mlpack.org/tutorial.html   <-- tutorials for mlpack
  http://www.mlpack.org/doxygen.php     <-- API documentation for mlpack
  http://www.mlpack.org/trac/           <-- development site for mlpack (Trac)

================================================================================

7. Bug reporting
   (see also http://www.mlpack.org/help.html )

If you find a bug in mlpack or have any problems, numerous routes are available
for help.

Trac is used for bug tracking, and can be found at http://www.mlpack.org/trac/.
It is easy to register an account and file a bug there, and the mlpack
development team will try to quickly resolve your issue.

In addition, mailing lists are available.  The mlpack discussion list is
available at

  https://lists.cc.gatech.edu/mailman/listinfo/mlpack

and the subversion commit list is available at

  https://lists.cc.gatech.edu/mailman/listinfo/mlpack-svn

Lastly, the IRC channel #mlpack on Freenode can be used to get help.

================================================================================
