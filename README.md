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
    <a href="https://www.mlpack.org/files/mlpack-4.5.1.tar.gz">current stable version (4.5.1)</a>
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
 - [Tutorials](doc/user/tutorials.md)
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
 3. [Installation](#3-installation)
 4. [Usage from C++](#4-usage-from-c)
     1. [Reducing compile time](#41-reducing-compile-time)
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

## 3. Installation

Detailed installation instructions can be found on the
[Installing mlpack](doc/user/install.md) page.

## 4. Usage from C++

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

***Warning:*** older versions of OpenBLAS (0.3.26 and older) compiled to use
pthreads may use too many threads for computation, causing significant slowdown.
OpenBLAS versions compiled with OpenMP do not suffer from this issue.  See the
[test build guide](doc/user/install.md#build-tests) for more details and simple
workarounds.

See also:

 * the [test program compilation section](doc/user/install.md#compiling-a-test-program)
   of the installation documentation,
 * the [C++ quickstart](doc/quickstart/cpp.md), and
 * the [examples repository](https://github.com/mlpack/examples) repository for
   some examples of mlpack applications in C++, with corresponding `Makefile`s.

### 4.1. Reducing compile time

mlpack is a template-heavy library, and if care is not used, compilation time of
a project can be very high.  Fortunately, there are a number of ways to reduce
compilation time:

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

## 5. Building mlpack's test suite

See the [installation instruction section](doc/user/install.md#build-tests).

## 6. Further Resources

More documentation is available for both users and developers.

 * [Documentation homepage](https://www.mlpack.org/doc/index.html)

To learn about the development goals of mlpack in the short- and medium-term
future, see the [vision document](https://www.mlpack.org/papers/vision.pdf).

If you have problems, find a bug, or need help, you can try visiting
the [mlpack help](https://www.mlpack.org/questions.html) page, or [mlpack on
Github](https://github.com/mlpack/mlpack/).  Alternately, mlpack help can be
found on Matrix at `#mlpack`; see also the
[community](https://www.mlpack.org/doc/developer/community.html) page.
