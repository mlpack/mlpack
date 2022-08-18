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
  <a href="https://dev.azure.com/mlpack/mlpack/_build?definitionId=1"><img alt="Azure DevOps builds (job)" src="https://img.shields.io/azure-devops/build/mlpack/84320e87-76e3-4b6e-8b6e-3adaf6b36eed/1/master?job=Linux&label=Linux%20Build&style=flat-square"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="License"></a>
  <a href="http://numfocus.org/donate-to-mlpack"><img src="https://img.shields.io/badge/sponsored%20by-NumFOCUS-orange.svg?style=flat-square&colorA=E1523D&colorB=007D8A" alt="NumFOCUS"></a>
</p>

<p align="center">
  <em>
    Download:
    <a href="https://www.mlpack.org/files/mlpack-3.4.2.tar.gz">current stable version (3.4.2)</a>
  </em>
</p>

**mlpack** is an intuitive, fast, and flexible header-only C++ machine learning
library with bindings to other languages.  It is meant to be a machine learning
analog to LAPACK, and aims to implement a wide array of machine learning methods
and functions as a "swiss army knife" for machine learning researchers.

mlpack's lightweight C++ implementation makes it ideal for deployment, and it
can also be used for interactive prototyping via C++ notebooks (these can be
seen in action on mlpack's [homepage](https://www.mlpack.org/)).

In addition to its powerful C++ interface, mlpack also provides command-line
programs, Python bindings, Julia bindings, Go bindings and R bindings.

***Quick links:***

 - Quickstart guides: [C++]( ), [CLI](doc/quickstart/cli.md),
   [Python](doc/quickstart/python.md), [R](doc/quickstart/R.md),
   [Julia](doc/quickstart/julia.md), [Go](doc/quickstart/go.md)
 - [mlpack homepage](https://www.mlpack.org/)
 - [mlpack documentation](https://www.mlpack.org/docs.html)
 - [Examples repository](https://github.com/mlpack/examples/)
 - [Tutorials](https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html)
 - [Development Site (Github)](https://www.github.com/mlpack/mlpack/)

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

## 0. Contents and Quick Links

 1. [Citation details](#1-citation-details)
 2. [Dependencies](#2-dependencies)
 3. [Installing and using mlpack in C++](#4-installing-and-using-mlpack-in-c++)
 4. [Building mlpack bindings to other languages](#5-building-mlpack-bindings-to-other-languages)
     1. [Command-line programs](#4i-command-line-programs)
     2. [Python bindings](#4ii-python-bindings)
     3. [R bindings](#4iii-r-bindings)
     4. [Julia bindings](#4iv-julia-bindings)
     5. [Go bindings](#4v-go-bindings)
 5. [Building mlpack's test suite](#5-building-mlpacks-test-suite)
 6. [Further resources](#6-further-resources)

## 1. Citation details

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

## 2. Dependencies

mlpack requires a C++14 compiler and has the following additional dependencies:

 - Armadillo      >= 9.800
 - ensmallen      >= 2.10.0
 - cereal         >= 1.1.2

If the STB library headers are available, image loading support will be
available.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

## 3. Installing and using mlpack in C++

Since mlpack is a header-only library, installing just the headers for use in a
C++ application is trivial.  From the root of the sources, configure and install
in the standard CMake way:

```sh
mkdir build && cd build/
cmake ../
sudo make install
```

You can add a few arguments to the `cmake` command to control the behavior of
the configuration and build process.  Simply add these to the `cmake` command.
Some options are given below:

 - `-DCMAKE_INSTALL_PREFIX=/install/root/` will set the root of the install
   directory to `/install/root` when `make install` is run.
 - `-DDOWNLOAD_DEPENDENCIES=ON` will automatically download mlpack's
   dependencies (ensmallen, Armadillo, and cereal).
 - `-DDEBUG=ON` will enable debugging symbols in any compiled bindings or tests.

There are also options to enable building bindings to each language that mlpack
supports; those are detailed in the following sections.

Once headers are installed with `make install`, using mlpack in an application
consists only of including it.  So, your program should include mlpack:

```c++
#include <mlpack.hpp>
```

and when you link, be sure to link against Armadillo.  If your example program
is `my_program.cpp`, your compiler is GCC, and you would like to compile with
OpenMP support (recommended) and optimizations, compile like this:

```sh
g++ -O3 -std=c++14 -o my_program my_program.cpp -larmadillo -fopenmp
```

See the [examples](https://github.com/mlpack/examples) repository for some
examples of mlpack applications in C++, with corresponding `Makefile`s.

## 4. Building mlpack bindings to other languages

mlpack is not just a header-only library: it also comes with bindings to a
number of other languages, this allows flexible use of mlpack's efficient
implementations from languages that aren't C++.

In general, you should *not* need to build these by hand---they should be
provided by either your system package manager or your language's package
manager.

Building the bindings for a particular language is done by calling `cmake` with
different options; each example below shows how to configure an individual set
of bindings, but it is of course possible to combine the options and build
bindings for many languages at once.

### 4.i. Command-line programs

The command-line programs have no extra dependencies.  The set of programs that
will be compiled is detailed and documented on the [command-line program
documentation page](https://www.mlpack.org/doc/stable/cli_documentation.html).

From the root of the mlpack sources, run the following commands to build and
install the command-line bindings:

```sh
mkdir build && cd build/
cmake -DBUILD_CLI_PROGRAMS=ON ../
make
sudo make install
```

You can use `make -j<N>`, where `N` is the number of cores on your machine, to
build in parallel; e.g., `make -j4` will use 4 cores to build.

### 4.ii. Python bindings

mlpack's Python bindings are available on
[PyPI](https://pypi.org/project/mlpack) and
[conda-forge](https://conda-forge.org/packages/mlpack), and can be installed
with either `pip install mlpack` or `conda install -c conda-forge mlpack`.
These sources are recommended, as building the Python bindings by hand can be
complex.

With that in mind, if you would still like to manually build the mlpack Python
bindings, first make sure that the following Python packages are installed:

 - setuptools
 - cython >= 0.24
 - numpy
 - pandas >= 0.15.0

Now, from the root of the mlpack sources, run the following commands to build
and install the Python bindings:

```sh
mkdir build && cd build/
cmake -DBUILD_PYTHON_BINDINGS=ON ../
make
sudo make install
```

You can use `make -j<N>`, where `N` is the number of cores on your machine, to
build in parallel; e.g., `make -j4` will use 4 cores to build.  You can also
specify a custom Python interpreter with the CMake option
`-DPYTHON_EXECUTABLE=/path/to/python`.

### 4.iii. R bindings

mlpack's R bindings are available as the R package
[mlpack](https://cran.r-project.org/web/packages/mlpack/index.html) on CRAN.
You can install the package by running `install.packages('mlpack')`, and this is
the recommended way of getting mlpack in R.

If you still wish to build the R bindings by hand, first make sure the following
dependencies are installed:

 - R >= 4.0
 - Rcpp >= 0.12.12
 - RcppArmadillo >= 0.9.800.0
 - RcppEnsmallen >= 0.2.10.0
 - roxygen2
 - testthat
 - pkgbuild

These can be installed with `install.packages()` inside of your R environment.
Once the dependencies are available, you can configure mlpack and build the R
bindings by running the following commands from the root of the mlpack sources:

```sh
mkdir build && cd build/
cmake -DBUILD_R_BINDINGS=ON ../
make
sudo make install
```

You may need to specify the location of the R program in the `cmake` command
with the option `-DR_EXECUTABLE=/path/to/R`.

Once the build is complete, a tarball can be found under the build directory in
`src/mlpack/bindings/R/`, and then that can be installed into your R environment
with a command like `install.packages(mlpack_3.4.3.tar.gz, repos=NULL,
type='source')`.

### 4.iv. Julia bindings

mlpack's Julia bindings are available by installing the
[mlpack.jl](https://github.com/mlpack/mlpack.jl) package using
`Pkg.add("mlpack.jl")`.  The process of building, packaging, and distributing
mlpack's Julia bindings is very nontrivial, so it is recommended to simply use
the version available in `Pkg`, but if you want to build the bindings by hand
anyway, you can configure and build them by running the following commands from
the root of the mlpack sources:

```sh
mkdir build && cd build/
cmake -DBUILD_JULIA_BINDINGS=ON ../
make
```

If CMake cannot find your Julia installation, you can add
`-DJULIA_EXECUTABLE=/path/to/julia` to the CMake configuration step.

Note that the `make install` step is not done above, since the Julia binding
build system was not meant to be installed directly.  Instead, to use handbuilt
bindings (for instance, to test them), one option is to start Julia with
`JULIA_PROJECT` set as an environment variable:

```sh
cd build/src/mlpack/bindings/julia/mlpack/
JULIA_PROJECT=$PWD julia
```

and then `using mlpack` should work.

### 4.v. Go bindings

To build mlpack's Go bindings, ensure that Go >= 1.11.0 is installed, and that
the Gonum package is available.
***TODO: how do you install these?***

Then, configuring and building the bindings can be done by running the following
commands from the root of the mlpack sources:

```sh
mkdir build && cd build/
cmake -DBUILD_GO_BINDINGS=ON ../
make
sudo make install
```

## 5. Building mlpack's test suite

mlpack contains an extensive test suite that exercises every part of the
codebase.  It is easy to build and run the tests with CMake and CTest, as below:

```sh
mkdir build && cd build/
cmake -DBUILD_TESTS=ON ../
make
ctest .
```

If you want to test the bindings, too, you will have to adapt the CMake
configuration command to turn on the language bindings that you want to
test---see the previous sections for details.

## 6. Further Resources



****
Tutorials to keep for users:

  formats.hpp (fine as-is)
  build_windows.hpp (needs adaptation)
  cv.hpp (as-is)
  hpt.hpp (as-is)
  sample_ml_app.hpp (pass through and adapt)

  needs earlier links:
    cli_quickstart.hpp
    go_quickstart.hpp
    julia_quickstart.hpp
    python_quickstart.hpp
    r_quickstart.hpp

Developer tutorials:

  timer.hpp
  version.hpp
  policies/
  bindings.hpp (but it's advanced)
  iodoc.hpp (also advanced, needs adaptation)

remove sample.hpp, and point instead towards examples/ repository
****

[mlpack on Github](https://www.github.com/mlpack/mlpack/)

Alternately, mlpack help can be found in IRC at `#mlpack` on chat.freenode.net.

If you wish to install mlpack to `/usr/local/include/mlpack/`, `/usr/local/lib/`,
and `/usr/local/bin/`, make sure you have root privileges (or write permissions
to those three directories), and simply type

    $ make install

You can now run the executables by name; the mlpack headers are found in
    `/usr/local/include/mlpack/`
and if Python bindings were built, you can access them with the `mlpack`
package in Python.

The documentation given here is only a fraction of the available documentation
for mlpack.  If doxygen is installed, you can type `make doc` to build the
documentation locally.  Alternately, up-to-date documentation is available for
older versions of mlpack:

  - [mlpack homepage](https://www.mlpack.org/)
  - [mlpack documentation](https://www.mlpack.org/docs.html)
  - [Tutorials](https://www.mlpack.org/doc/mlpack-git/doxygen/tutorials.html)
  - [Development Site (Github)](https://www.github.com/mlpack/mlpack/)
  - [API documentation (Doxygen)](https://www.mlpack.org/doc/mlpack-git/doxygen/index.html)

To learn about the development goals of mlpack in the short- and medium-term
future, see the [vision document](https://www.mlpack.org/papers/vision.pdf).

   (see also [mlpack help](https://www.mlpack.org/questions.html))

If you find a bug in mlpack or have any problems, numerous routes are available
for help.

Github is used for bug tracking, and can be found at
https://github.com/mlpack/mlpack/issues.
It is easy to register an account and file a bug there, and the mlpack
development team will try to quickly resolve your issue.

In addition, mailing lists are available.  The mlpack discussion list is
available at

  [mlpack discussion list](http://lists.mlpack.org/mailman/listinfo/mlpack)

and the git commit list is available at

  [commit list](http://lists.mlpack.org/mailman/listinfo/mlpack-git)

Lastly, the IRC channel `#mlpack` on Freenode can be used to get help.
