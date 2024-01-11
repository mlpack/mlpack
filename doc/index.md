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

## mlpack basics

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

## mlpack algorithm documentation

Documentation for each machine learning algorithm that mlpack implements is
detailed in the sections below.

 * [Classification algorithms](#classification-algorithms): classify points as
   discrete labels (`0`, `1`, `2`, ...).
 * [Regression algorithms](#regression-algorithms): predict continuous values.
 * [Clustering algorithms](#clustering-algorithms): group points into clusters.
 * [Geometric algorithms](#geometric-algorithms): computations based on distance
   metrics (nearest neighbors, kernel density estimation, etc.).
 * [Preprocessing utilities](#preprocessing-utilities): prepare data for machine
   learning algorithms.
 * [Transformations](#transformations): transform data from one space to
   another (principal components analysis, etc.).
 * [Modeling utilities](#modeling-utilities): cross-validation, hyperparameter
   tuning, etc.

## Bindings to other languages

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

## Examples and further documentation

 * [mlpack examples repository](https://github.com/mlpack/examples/): numerous
   fully-working example applications of mlpack, in C++ and other languages.
 * [mlpack models repository](https://github.com/mlpack/models/): complex models
   in C++ built with mlpack

For additional documentation beyond what is covered in all the resources above,
the source code should be consulted.  Each method is fully documented.

## Developer documentation

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

## Algorithm documentation

### Classification algorithms

Classify points as discrete labels (`0`, `1`, `2`, ...).

 * [`AdaBoost`](user/methods/adaboost.md): Adaptive Boosting
 * [`DecisionTree`](user/methods/decision_tree.md): ID3-style decision tree
   classifier
 * [`LogisticRegression`](user/methods/logistic_regression.md): L2-regularized
   logistic regression (two-class only)
 * [`Perceptron`](user/methods/perceptron.md): simple Perceptron classifier
 * [`SoftmaxRegression`](user/methods/softmax_regression.md): L2-regularized
   softmax regression (i.e. multi-class logistic regression)

### Regression algorithms

Predict continuous values.

 * [`DecisionTreeRegressor`](user/methods/decision_tree_regressor.md): ID3-style
   decision tree regressor

### Clustering algorithms

Group points into clusters.

<!-- TODO: add some -->

### Geometric algorithms

Computations based on distance metrics.

<!-- TODO: add some -->

### Preprocessing utilities

Prepare data for machine learning algorithms.

<!-- TODO: add some -->

### Transformations

Transform data from one space to another.

<!-- TODO: add some -->

### Modeling utilities

Cross-validation, hyperparameter tuning, etc.

<!-- TODO: add some -->
