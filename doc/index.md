# Documentation for mlpack
## A fast, flexible machine learning library

mlpack is an intuitive, fast, and flexible header-only C++ machine learning
library with bindings to other languages.  It aims to provide fast, lightweight
implementations of both common and cutting-edge machine learning algorithms.

mlpack's lightweight C++ implementation makes it ideal for deployment, and it
can also be used for interactive prototyping via C++ notebooks (these can be
seen in action on mlpack's [homepage](https://www.mlpack.org/)).

In addition to its powerful C++ interface, mlpack also provides command-line
programs, and bindings to the Python, R, Julia, and Go languages.

_If you use mlpack, please [cite the software](citation.md)._

### mlpack basics

Installing mlpack can be done using the
[instructions in the README](README.md#3-installing-and-using-mlpack-in-c);
or the [Windows build guide](user/build_windows.md).  Then, the following
simple guides are good places to get started:

 * [mlpack C++ quickstart](quickstart/cpp.md): create a couple simple C++
   programs that use mlpack
 * [Sample Windows mlpack C++ application](user/sample_ml_app.md): create a
   working mlpack Windows program using Visual Studio

After that, it's a good idea to familiarize yourself with the basics of the
library.  The documentation for mlpack's algorithms depends on the concepts in
the pages below.

 * [Matrices and data in mlpack](user/matrices.md)
 * [Loading and saving mlpack objects](user/load_save.md)
 * [Core mlpack documentation](user/core.md): reference documentation for all
   core classes and functions that are used in mlpack.

### mlpack algorithm documentation

Documentation for each machine learning algorithm that mlpack implements is
detailed in the pages below.

 * [Classification algorithms](user/classification.md): classify points as
   discrete labels (`0`, `1`, `2`, ...).
 * [Regression algorithms](user/regression.md): predict continuous values.
 * [Clustering algorithms](user/clustering.md): group points into clusters.
 * [Geometric algorithms](user/geometry.md): computations based on distance
   metrics (nearest neighbors, kernel density estimation, etc.).
 * [Preprocessing utilities](user/preprocessing.md): prepare data for machine
   learning algorithms.
 * [Transformations](user/transformations.md): transform data from one space to
   another (principal components analysis, etc.).
 * [Modeling utilities](user/modeling.md): cross-validation, hyperparameter
   tuning, etc.

### Bindings to other languages

mlpack's bindings to other languages have less complete functionality than
mlpack in C++, but almost all of the same algorithms are available.

***Python***:

 * [Python quickstart](quickstart/python.md)
 * [Python reference documentation](https://www.mlpack.org/doc/python_documentation.html)

***Julia***:

 * [Julia quickstart](quickstart/julia.md)
 * [Julia reference documentation](https://www.mlpack.org/doc/julia_documentation.html)

***R***:

 * [R quickstart](quickstart/r.md)
 * [R reference documentation](https://www.mlpack.org/doc/r_documentation.html)

***Command-line programs***:

 * [Command-line quickstart](quickstart/cli.md)
 * [Command-line reference documentation](https://www.mlpack.org/doc/cli_documentation.html)

***Go***:

 * [Go quickstart](quickstart/go.md)
 * [Go reference documentation](https://www.mlpack.org/doc/go_documentation.html)

### Examples and further documentation

 * [mlpack examples repository](https://github.com/mlpack/examples/): numerous
   fully-working example applications of mlpack, in C++ and other languages.
 * [mlpack models repository](https://github.com/mlpack/models/): complex models
   in C++ built with mlpack

For additional documentation beyond what is covered in all the resources above,
the source code should be consulted.  Each method is fully documented.

### Developer documentation

Throughout the codebase, mlpack uses some common template parameter policies.
These are documented below.

 * [The `ElemType` policy](developer/elemtype.md): element types for data
 * [The `MetricType` policy](developer/metrics.md): distance metrics
 * [The `KernelType` policy](developer/kernels.md): kernel functions
 * [The `TreeType` policy](developer/trees.md): space trees (ball trees,
   KD-trees, etc.)

In addition, the following documentation may be useful when developing bindings
for other languages:

 * [Timers](developer/timer.md): timing parts of bindings
 * [Writing an mlpack binding](developer/iodoc.md): simple examples of mlpack
   bindings
 * [Automatic bindings](developer/bindings.md): details on mlpack's automatic
   binding generator system.

