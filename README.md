<h2 align="center">
  <a href="https://mlpack.org"><img src="https://cdn.jsdelivr.net/gh/mlpack/mlpack.org@e7d36ed8/mlpack-black.svg" style="background-color:rgba(0,0,0,0);" height=230 alt="mlpack: a fast, header-only machine learning library"></a>
  <br>a fast, header-only machine learning library<br>
</h2>

<h5 align="center">
  <a href="https://mlpack.org">Home</a> |
  <a href="https://www.mlpack.org/download.html">Download</a> |
  <a href="https://www.mlpack.org/doc/index.html">Documentation</a> |
  <a href="https://www.mlpack.org/questions.html">Help</a> |
</h5>

<p align="center">
  <a href="https://dev.azure.com/mlpack/mlpack/_build?definitionId=1"><img alt="Azure DevOps builds (job)" src="https://img.shields.io/azure-devops/build/mlpack/84320e87-76e3-4b6e-8b6e-3adaf6b36eed/1/master?job=Linux&label=Linux%20Build&style=flat-square"></a>
  <a href="https://opensource.org/license/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://numfocus.org/donate-to-mlpack"><img src="https://img.shields.io/badge/sponsored%20by-NumFOCUS-orange.svg?style=flat-square&colorA=E1523D&colorB=007D8A" alt="NumFOCUS"></a>
</p>

<p align="center">
  <em>
    Download:
    <a href="https://www.mlpack.org/files/mlpack-4.4.0.tar.gz">current stable version (4.4.0)</a>
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

 - Quickstart guides: [C++](doc/quickstart/cpp.md),
   [CLI](doc/quickstart/cli.md), [Python](doc/quickstart/python.md),
   [R](doc/quickstart/r.md), [Julia](doc/quickstart/julia.md),
   [Go](doc/quickstart/go.md)
 - [mlpack homepage](https://www.mlpack.org/)
 - [mlpack documentation](https://www.mlpack.org/doc/index.html)
 - [Examples repository](https://github.com/mlpack/examples/)
 - [Tutorials](doc/tutorials/README.md)
 - [Development Site (Github)](https://github.com/mlpack/mlpack/)

[//]: # (numfocus-fiscal-sponsor-attribution)

mlpack uses an [open governance model](./GOVERNANCE.md) and is fiscally
sponsored by [NumFOCUS](https://numfocus.org/).  Consider making a
[tax-deductible donation](https://numfocus.org/donate-to-mlpack) to help the
project pay for developer time, professional services, travel, workshops, and a
variety of other needs.

<div align="center">
  <a href="https://numfocus.org/">
    <img height="60"
         src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
         align="middle"
         alt="NumFOCUS logo">
  </a>
</div>
<br>

## 0. Contents

 1. [Citation details](#1-citation-details)
 2. [Dependencies](#2-dependencies)
 3. [Installing and using mlpack in C++](#3-installing-and-using-mlpack-in-c)
 4. [Building mlpack bindings to other languages](#4-building-mlpack-bindings-to-other-languages)
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

    @article{mlpack2023,
        title     = {mlpack 4: a fast, header-only C++ machine learning library},
        author    = {Ryan R. Curtin and Marcus Edel and Omar Shrit and 
                     Shubham Agrawal and Suryoday Basak and James J. Balamuta and 
                     Ryan Birmingham and Kartik Dutt and Dirk Eddelbuettel and 
                     Rishabh Garg and Shikhar Jaiswal and Aakash Kaushik and 
                     Sangyeon Kim and Anjishnu Mukherjee and Nanubala Gnana Sai and 
                     Nippun Sharma and Yashwant Singh Parihar and Roshan Swain and 
                     Conrad Sanderson},
        journal   = {Journal of Open Source Software},
        volume    = {8},
        number    = {82},
        pages     = {5026},
        year      = {2023},
        doi       = {10.21105/joss.05026},
        url       = {https://doi.org/10.21105/joss.05026}
    }

Citations are beneficial for the growth and improvement of mlpack.

## 2. Dependencies

**mlpack** requires the following additional dependencies:
 - C++17 compiler
 - [Armadillo](https://arma.sourceforge.net)      &nbsp;&emsp;>= 10.8
 - [ensmallen](https://ensmallen.org)      &emsp;>= 2.10.0
 - [cereal](http://uscilab.github.io/cereal/)         &ensp;&nbsp;&emsp;&emsp;>= 1.1.2

If the STB library headers are available, image loading support will be
available.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

## 3. Installing and using mlpack in C++

*See also the [C++ quickstart](doc/quickstart/cpp.md).*

Since mlpack is a header-only library, installing just the headers for use in a
C++ application is trivial.

From the root of the sources, configure and install
in the standard CMake way:

```sh
mkdir build && cd build/
cmake ..
sudo make install
```

If the `cmake ..` command fails due to unavailable dependencies, consider either using the
`-DDOWNLOAD_DEPENDENCIES=ON` option as detailed in [the following
subsection](#31-additional-build-options), or ensure that mlpack's dependencies
are installed, e.g. using the system package manager.  For example, on Debian
and Ubuntu, all relevant dependencies can be installed with `sudo apt-get
install libarmadillo-dev libensmallen-dev libcereal-dev libstb-dev g++ cmake`.

Alternatively, since CMake v3.14.0 the `cmake` command can create the build
folder itself, and so the above commands can be rewritten as follows:

```sh
cmake -S . -B build
sudo cmake --build build --target install
```

During configuration, CMake adjusts the file `mlpack/config.hpp` using the
details of the local system.  This file can be modified by hand as necessary
before or after installation.

### 3.1. Additional build options

You can add a few arguments to the `cmake` command to control the behavior of
the configuration and build process.  Simply add these to the `cmake` command.
Some options are given below:

 - `-DDOWNLOAD_DEPENDENCIES=ON` will automatically download mlpack's
   dependencies (ensmallen, Armadillo, and cereal).  Installing Armadillo this
   way is not recommended and it is better to use your system package manager
   when possible (see [below](#31a-linking-with-autodownloaded-armadillo)).
 - `-DCMAKE_INSTALL_PREFIX=/install/root/` will set the root of the install
   directory to `/install/root` when `make install` is run.
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
g++ -O3 -std=c++17 -o my_program my_program.cpp -larmadillo -fopenmp
```

Note that if you want to serialize (save or load) neural networks, you should
add `#define MLPACK_ENABLE_ANN_SERIALIZATION` before including `<mlpack.hpp>`.
If you don't define `MLPACK_ENABLE_ANN_SERIALIZATION` and your code serializes a
neural network, a compilation error will occur.

See the [C++ quickstart](doc/quickstart/cpp.md) and the
[examples](https://github.com/mlpack/examples) repository for some examples
of mlpack applications in C++, with corresponding `Makefile`s.

#### 3.1.a. Linking with autodownloaded Armadillo

When the autodownloader is used to download Armadillo
(`-DDOWNLOAD_DEPENDENCIES=ON`), the Armadillo runtime library is not built and
Armadillo must be used in header-only mode.  The autodownloader also does not
download dependencies of Armadillo such as OpenBLAS.  For this reason, it is
recommended to instead install Armadillo using your system package manager,
which will also install the dependencies of Armadillo.  For example, on Ubuntu
and Debian systems, Armadillo can be installed with

```sh
sudo apt-get install libarmadillo-dev
```

and other package managers such as `dnf` and `brew` and `pacman` also have
Armadillo packages available.

If the autodownloader is used to provide Armadillo, mlpack programs cannot be
linked with `-larmadillo`.  Instead, you must link directly with the
dependencies of Armadillo.  For example, on a system that has OpenBLAS
available, compilation can be done like this:

```sh
g++ -O3 -std=c++17 -o my_program my_program.cpp -lopenblas -fopenmp
```

See [the Armadillo documentation](https://arma.sourceforge.net/faq.html#linking)
for more information on linking Armadillo programs.

### 3.2. Reducing compile time

mlpack is a template-heavy library, and if care is not used, compilation time of
a project can be increased greatly.  Fortunately, there are a number of ways to
reduce compilation time:

 * Include individual headers, like `<mlpack/methods/decision_tree.hpp>`, if you
   are only using one component, instead of `<mlpack.hpp>`.  This reduces the
   amount of work the compiler has to do.

 * Only use the `MLPACK_ENABLE_ANN_SERIALIZATION` definition if you are
   serializing neural networks in your code.  When this define is enabled,
   compilation time will increase significantly, as the compiler must generate
   code for every possible type of layer.  (The large amount of extra
   compilation overhead is why this is not enabled by default.)

 * If you are using mlpack in multiple .cpp files, consider using [`extern
   templates`](https://isocpp.org/wiki/faq/cpp11-language-templates) so that the
   compiler only instantiates each template once; add an explicit template
   instantiation for each mlpack template type you want to use in a .cpp file,
   and then use `extern` definitions elsewhere to let the compiler know it
   exists in a different file.

Other strategies exist too, such as precompiled headers, compiler options,
[`ccache`](https://ccache.dev), and others.

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

*See also the [command-line quickstart](doc/quickstart/cli.md).*

The command-line programs have no extra dependencies.  The set of programs that
will be compiled is detailed and documented on the [command-line program
documentation page](doc/user/bindings/cli.md).

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

*See also the [Python quickstart](doc/quickstart/python.md).*

mlpack's Python bindings are available on
[PyPI](https://pypi.org/project/mlpack/) and
[conda-forge](https://anaconda.org/conda-forge/mlpack), and can be installed
with either `pip install mlpack` or `conda install -c conda-forge mlpack`.
These sources are recommended, as building the Python bindings by hand can be
complex.

With that in mind, if you would still like to manually build the mlpack Python
bindings, first make sure that the following Python packages are installed:

 - setuptools
 - wheel
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

*See also the [R quickstart](doc/quickstart/r.md).*

mlpack's R bindings are available as the R package
[mlpack](https://cran.r-project.org/web/packages/mlpack/index.html) on CRAN.
You can install the package by running `install.packages('mlpack')`, and this is
the recommended way of getting mlpack in R.

If you still wish to build the R bindings by hand, first make sure the following
dependencies are installed:

 - R >= 4.0
 - Rcpp >= 0.12.12
 - RcppArmadillo >= 0.10.8.0
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

*See also the [Julia quickstart](doc/quickstart/julia.md).*

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

*See also the [Go quickstart](doc/quickstart/go.md).*

To build mlpack's Go bindings, ensure that Go >= 1.11.0 is installed, and that
the Gonum package is available.  You can use `go get` to install mlpack for Go:

```sh
go get -u -d mlpack.org/v1/mlpack
cd ${GOPATH}/src/mlpack.org/v1/mlpack
make install
```

The process of building the Go bindings by hand is a little tedious, so
following the steps above is recommended.  However, if you wish to build the Go
bindings by hand anyway, you can do this by running the following commands from
the root of the mlpack sources:

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

More documentation is available for both users and developers.

***User documentation***:

 - [Matrices in mlpack](doc/user/matrices.md)
 - [Loading and saving mlpack objects](doc/user/load_save.md)
 - [Cross-Validation](doc/user/cv.md)
 - [Hyper-parameter Tuning](doc/user/hpt.md)
 - [Building mlpack from source on Windows](doc/user/build_windows.md)
 - [Sample C++ ML App for Windows](doc/user/sample_ml_app.md)
 - [mlpack core library documentation](doc/user/core.md)
 - [Examples repository](https://github.com/mlpack/examples/)

***Tutorials:***

 - [Alternating Matrix Factorization (AMF)](doc/tutorials/amf.md)
 - [Artificial Neural Networks (ANN)](doc/tutorials/ann.md)
 - [Approximate k-Furthest Neighbor Search (`approx_kfn`)](doc/tutorials/approx_kfn.md)
 - [Collaborative Filtering (CF)](doc/tutorials/cf.md)
 - [DatasetMapper](doc/tutorials/datasetmapper.md)
 - [Density Estimation Trees (DET)](doc/tutorials/det.md)
 - [Euclidean Minimum Spanning Trees (EMST)](doc/tutorials/emst.md)
 - [Fast Max-Kernel Search (FastMKS)](doc/tutorials/fastmks.md)
 - [Image Utilities](doc/tutorials/image.md)
 - [k-Means Clustering](doc/tutorials/kmeans.md)
 - [Linear Regression](doc/tutorials/linear_regression.md)
 - [Neighbor Search (k-Nearest-Neighbors)](doc/tutorials/neighbor_search.md)
 - [Range Search](doc/tutorials/range_search.md)
 - [Reinforcement Learning](doc/tutorials/reinforcement_learning.md)

***Developer documentation***:

 - [Writing an mlpack binding](doc/developer/iodoc.md)
 - [mlpack Timers](doc/developer/timer.md)
 - [mlpack automatic bindings to other languages](doc/developer/bindings.md)
 - [The ElemType policy in mlpack](doc/developer/elemtype.md)
 - [The KernelType policy in mlpack](doc/developer/kernels.md)
 - [The DistanceType policy in mlpack](doc/developer/distances.md)
 - [The TreeType policy in mlpack](doc/developer/trees.md)

To learn about the development goals of mlpack in the short- and medium-term
future, see the [vision document](https://www.mlpack.org/papers/vision.pdf).

If you have problems, find a bug, or need help, you can try visiting
the [mlpack help](https://www.mlpack.org/questions.html) page, or [mlpack on
Github](https://github.com/mlpack/mlpack/).  Alternately, mlpack help can be
found on Matrix at `#mlpack`; see also the
[community](https://www.mlpack.org/doc/developer/community.html) page.
