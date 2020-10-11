<h2 align="center">
  <a href="http://mlpack.org"><img
src="https://cdn.rawgit.com/mlpack/mlpack.org/e7d36ed8/mlpack-black.svg" style="background-color:rgba(0,0,0,0);" height=230 alt="mlpack: a fast, flexible machine learning library"></a>
  <br>a fast, flexible machine learning library<br>
</h2>

<h5 align="center">
  <a href="https://mlpack.org">Home</a> |
  <a href="https://www.mlpack.org/docs.html">Documentation</a> |
  <a href="https://www.mlpack.org/doc/mlpack-git/doxygen/index.html">Doxygen</a> |
  <a href="https://www.mlpack.org/community.html">Community</a> |
  <a href="https://www.mlpack.org/questions.html">Help</a> |
  <a href="https://webchat.freenode.net/?channels=mlpack">IRC Chat</a>
</h5>

<p align="center">
  <a href="http://ci.mlpack.org/job/mlpack%20-%20git%20commit%20test/"><img src="https://img.shields.io/jenkins/build.svg?jobUrl=http%3A%2F%2Fci.mlpack.org%2Fjob%2Fmlpack%2520-%2520git%2520commit%2520test%2F&label=Linux%20build&style=flat-square" alt="Jenkins"></a>
  <a href="https://coveralls.io/github/mlpack/mlpack?branch=master"><img src="https://img.shields.io/coveralls/mlpack/mlpack/master.svg?style=flat-square" alt="Coveralls"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="License"></a>
  <a href="http://numfocus.org/donate-to-mlpack"><img src="https://img.shields.io/badge/sponsored%20by-NumFOCUS-orange.svg?style=flat-square&colorA=E1523D&colorB=007D8A" alt="NumFOCUS"></a>
</p>

<p align="center">
  <em>
    Download:
    <a href="https://www.mlpack.org/files/mlpack-3.4.1.tar.gz">current stable version (3.4.1)</a>
  </em>
</p>

**mlpack** is an intuitive, fast, and flexible C++ machine learning library with
bindings to other languages.  It is meant to be a machine learning analog to
LAPACK, and aims to implement a wide array of machine learning methods and
functions as a "swiss army knife" for machine learning researchers.  In addition
to its powerful C++ interface, mlpack also provides command-line programs,
Python bindings, Julia bindings, Go bindings and R bindings.

[//]: # (numfocus-fiscal-sponsor-attribution)

mlpack uses an [open governance model](./GOVERNANCE.md) and is fiscally
sponsored by [NumFOCUS](https://numfocus.org/).  Consider making a
[tax-deductible donation](https://numfocus.org/donate-to-mlpack) to help the
project pay for developer time, professional services, travel, workshops, and a
variety of other needs.

<div align="center">
  <a href="https://numfocus.org/">
    <img height="60px"
         src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
         align="center">
  </a>
</div>
<br>

### 0. Contents

  1. [Introduction](#1-introduction)
  2. [Citation details](#2-citation-details)
  3. [Dependencies](#3-dependencies)
  4. [Building mlpack from source](#4-building-mlpack-from-source)
  5. [Running mlpack programs](#5-running-mlpack-programs)
  6. [Using mlpack from Python](#6-using-mlpack-from-python)
  7. [Further documentation](#7-further-documentation)
  8. [Bug reporting](#8-bug-reporting)

###  1. Introduction

The mlpack website can be found at https://www.mlpack.org and it contains
numerous tutorials and extensive documentation.  This README serves as a guide
for what mlpack is, how to install it, how to run it, and where to find more
documentation. The website should be consulted for further information:

  - [mlpack homepage](https://www.mlpack.org/)
  - [mlpack documentation](https://www.mlpack.org/docs.html)
  - [Tutorials](https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html)
  - [Development Site (Github)](https://www.github.com/mlpack/mlpack/)
  - [API documentation (Doxygen)](https://www.mlpack.org/doc/mlpack-git/doxygen/index.html)

### 2. Citation details

If you use mlpack in your research or software, please cite mlpack using the
citation below (given in BibTeX format):

    @article{mlpack2018,
        title     = {mlpack 3: a fast, flexible machine learning library},
        author    = {Curtin, Ryan R. and Edel, Marcus and Lozhnikov, Mikhail and
                     Mentekidis, Yannis and Ghaisas, Sumedh and Zhang,
                     Shangtong},
        journal   = {Journal of Open Source Software},
        volume    = {3},
        issue     = {26},
        pages     = {726},
        year      = {2018},
        doi       = {10.21105/joss.00726},
        url       = {https://doi.org/10.21105/joss.00726}
    }

Citations are beneficial for the growth and improvement of mlpack.

### 3. Dependencies

mlpack has the following dependencies:

      Armadillo      >= 8.400.0
      Boost (math_c99, unit_test_framework, serialization,
             spirit) >= 1.58.0
      CMake          >= 3.2.2
      ensmallen      >= 2.10.0

All of those should be available in your distribution's package manager.  If
not, you will have to compile each of them by hand.  See the documentation for
each of those packages for more information.

If you would like to use or build the mlpack Python bindings, make sure that the
following Python packages are installed:

      setuptools
      cython >= 0.24
      numpy
      pandas >= 0.15.0

If you would like to build the Julia bindings, make sure that Julia >= 1.3.0 is
installed.

If you would like to build the Go bindings, make sure that Go >= 1.11.0 is
installed with this package:

     Gonum

If you would like to build the R bindings, make sure that R >= 4.0 is
installed with these R packages.

     Rcpp >= 0.12.12
     RcppArmadillo >= 0.8.400.0
     RcppEnsmallen >= 0.2.10.0
     BH >= 1.58
     roxygen2

If the STB library headers are available, image loading support will be
compiled.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

### 4. Building mlpack from source

This document discusses how to build mlpack from source. These build directions 
will work for any Linux-like shell environment (for example Ubuntu, macOS,
FreeBSD etc). However, mlpack is in the repositories of many Linux distributions 
and so it may be easier to use the package manager for your system.  For example, 
on Ubuntu, you can install mlpack with the following command:

    $ sudo apt-get install libmlpack-dev

Note: Older Ubuntu versions may not have the most recent version of mlpack
available---for instance, at the time of this writing, Ubuntu 16.04 only has
mlpack 3.4.1 available.  Options include upgrading your Ubuntu version, finding
a PPA or other non-official sources, or installing with a manual build.

There are some useful pages to consult in addition to this section:

  - [Building mlpack From Source](https://www.mlpack.org/doc/mlpack-git/doxygen/build.html)
  - [Building mlpack From Source on Windows](https://www.mlpack.org/doc/mlpack-git/doxygen/build_windows.html)

mlpack uses CMake as a build system and allows several flexible build
configuration options. You can consult any of the CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed.

First, unpack the mlpack source and change into the unpacked directory.  Here we
use mlpack-x.y.z where x.y.z is the version.

    $ tar -xzf mlpack-x.y.z.tar.gz
    $ cd mlpack-x.y.z

Then, make a build directory.  The directory can have any name, but 'build' is
sufficient.

    $ mkdir build
    $ cd build

The next step is to run CMake to configure the project.  Running CMake is the
equivalent to running `./configure` with autotools. If you run CMake with no
options, it will configure the project to build with no debugging symbols and 
no profiling information:

    $ cmake ../

Options can be specified to compile with debugging information and profiling information:

    $ cmake -D DEBUG=ON -D PROFILE=ON ../

Options are specified with the -D flag.  The allowed options include:

    DEBUG=(ON/OFF): compile with debugging symbols
    PROFILE=(ON/OFF): compile with profiling symbols
    ARMA_EXTRA_DEBUG=(ON/OFF): compile with extra Armadillo debugging symbols
    BOOST_ROOT=(/path/to/boost/): path to root of boost installation
    ARMADILLO_INCLUDE_DIR=(/path/to/armadillo/include/): path to Armadillo headers
    ARMADILLO_LIBRARY=(/path/to/armadillo/libarmadillo.so): Armadillo library
    BUILD_CLI_EXECUTABLES=(ON/OFF): whether or not to build command-line programs
    BUILD_PYTHON_BINDINGS=(ON/OFF): whether or not to build Python bindings
    PYTHON_EXECUTABLE=(/path/to/python_version): Path to specific Python executable
    BUILD_JULIA_BINDINGS=(ON/OFF): whether or not to build Julia bindings
    JULIA_EXECUTABLE=(/path/to/julia): Path to specific Julia executable
    BUILD_GO_BINDINGS=(ON/OFF): whether or not to build Go bindings
    GO_EXECUTABLE=(/path/to/go): Path to specific Go executable
    BUILD_GO_SHLIB=(ON/OFF): whether or not to build shared libraries required by Go bindings
    BUILD_R_BINDINGS=(ON/OFF): whether or not to build R bindings
    R_EXECUTABLE=(/path/to/R): Path to specific R executable
    BUILD_TESTS=(ON/OFF): whether or not to build tests
    BUILD_SHARED_LIBS=(ON/OFF): compile shared libraries as opposed to
       static libraries
    DISABLE_DOWNLOADS=(ON/OFF): whether to disable all downloads during build
    DOWNLOAD_ENSMALLEN=(ON/OFF): If ensmallen is not found, download it
    ENSMALLEN_INCLUDE_DIR=(/path/to/ensmallen/include): path to include directory
       for ensmallen
    DOWNLOAD_STB_IMAGE=(ON/OFF): If STB is not found, download it
    STB_IMAGE_INCLUDE_DIR=(/path/to/stb/include): path to include directory for
       STB image library
    USE_OPENMP=(ON/OFF): whether or not to use OpenMP if available

Other tools can also be used to configure CMake, but those are not documented
here.  See [this section of the build guide](https://www.mlpack.org/doc/mlpack-git/doxygen/build.html#build_config)
for more details, including a full list of options, and their default values.

By default, command-line programs will be built, and if the Python dependencies
(Cython, setuptools, numpy, pandas) are available, then Python bindings will
also be built.  OpenMP will be used for parallelization when possible by
default.

Once CMake is configured, building the library is as simple as typing 'make'.
This will build all library components as well as 'mlpack_test'.

    $ make

If you do not want to build everything in the library, individual components 
of the build can be specified:

    $ make mlpack_pca mlpack_knn mlpack_kfn

If the build fails and you cannot figure out why, register an account on Github
and submit an issue. The mlpack developers will quickly help you figure it out:

[mlpack on Github](https://www.github.com/mlpack/mlpack/)

Alternately, mlpack help can be found in IRC at `#mlpack` on chat.freenode.net.

If you wish to install mlpack to `/usr/local/include/mlpack/`, `/usr/local/lib/`,
and `/usr/local/bin/`, make sure you have root privileges (or write permissions 
to those three directories), and simply type

    $ make install

You can now run the executables by name; you can link against mlpack with
    `-lmlpack`
and the mlpack headers are found in
    `/usr/local/include/mlpack/`
and if Python bindings were built, you can access them with the `mlpack`
package in Python.

If running the programs (i.e. `$ mlpack_knn -h`) gives an error of the form

    error while loading shared libraries: libmlpack.so.2: cannot open shared object file: No such file or directory

then be sure that the runtime linker is searching the directory where
`libmlpack.so` was installed (probably `/usr/local/lib/` unless you set it
manually).  One way to do this, on Linux, is to ensure that the
`LD_LIBRARY_PATH` environment variable has the directory that contains
`libmlpack.so`.  Using bash, this can be set easily:

    export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"

(or whatever directory `libmlpack.so` is installed in.)

### 5. Running mlpack programs

After building mlpack, the executables will reside in `build/bin/`.  You can call
them from there, or you can install the library and (depending on system
settings) they should be added to your PATH and you can call them directly.  The
documentation below assumes the executables are in your PATH.

Consider the 'mlpack_knn' program, which finds the k nearest neighbors in a
reference dataset of all the points in a query set.  That is, we have a query
and a reference dataset. For each point in the query dataset, we wish to know
the k points in the reference dataset which are closest to the given query
point.

Alternately, if the query and reference datasets are the same, the problem can
be stated more simply: for each point in the dataset, we wish to know the k
nearest points to that point.

Each mlpack program has extensive help documentation which details what the
method does, what each of the parameters is, and how to use them:

```shell
$ mlpack_knn --help
```

Running `mlpack_knn` on one dataset (that is, the query and reference
datasets are the same) and finding the 5 nearest neighbors is very simple:

```shell
$ mlpack_knn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 5 -v
```

The `-v (--verbose)` flag is optional; it gives informational output.  It is not
unique to `mlpack_knn` but is available in all mlpack programs.  Verbose
output also gives timing output at the end of the program, which can be very
useful.

### 6. Using mlpack from Python

If mlpack is installed to the system, then the mlpack Python bindings should be
automatically in your PYTHONPATH, and importing mlpack functionality into Python
should be very simple:

```python
>>> from mlpack import knn
```

Accessing help is easy:

```python
>>> help(knn)
```

The API is similar to the command-line programs.  So, running `knn()`
(k-nearest-neighbor search) on the numpy matrix `dataset` and finding the 5
nearest neighbors is very simple:

```python
>>> output = knn(reference=dataset, k=5, verbose=True)
```

This will store the output neighbors in `output['neighbors']` and the output
distances in `output['distances']`.  Other mlpack bindings function similarly,
and the input/output parameters exactly match those of the command-line
programs.

### 7. Further documentation

The documentation given here is only a fraction of the available documentation
for mlpack.  If doxygen is installed, you can type `make doc` to build the
documentation locally.  Alternately, up-to-date documentation is available for
older versions of mlpack:

  - [mlpack homepage](https://www.mlpack.org/)
  - [mlpack documentation](https://www.mlpack.org/docs.html)
  - [Tutorials](https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html)
  - [Development Site (Github)](https://www.github.com/mlpack/mlpack/)
  - [API documentation (Doxygen)](https://www.mlpack.org/doc/mlpack-git/doxygen/index.html)

### 8. Bug reporting

   (see also [mlpack help](https://www.mlpack.org/questions.html))

If you find a bug in mlpack or have any problems, numerous routes are available
for help.

Github is used for bug tracking, and can be found at
https://github.com/mlpack/mlpack/.
It is easy to register an account and file a bug there, and the mlpack
development team will try to quickly resolve your issue.

In addition, mailing lists are available.  The mlpack discussion list is
available at

  [mlpack discussion list](http://lists.mlpack.org/mailman/listinfo/mlpack)

and the git commit list is available at

  [commit list](http://lists.mlpack.org/mailman/listinfo/mlpack-git)

Lastly, the IRC channel `#mlpack` on Freenode can be used to get help.
