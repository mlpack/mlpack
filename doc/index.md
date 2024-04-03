# Documentation for mlpack

<!--
This file contains the landing page for mlpack documentation.  Note that if you
change any section headers, or add any new algorithms, the sidebar in
sidebar.html will need to be manually modified!
-->

## A fast, flexible machine learning library

mlpack is an intuitive, fast, and flexible header-only C++ machine learning
library with bindings to other languages.  It aims to provide fast, lightweight
implementations of both common and cutting-edge machine learning algorithms.

mlpack's lightweight C++ implementation makes it ideal for deployment, and it
can also be used for interactive prototyping via C++ notebooks (these can be
seen in action on mlpack's [homepage](https://www.mlpack.org/)).

In addition to its [powerful C++ interface](quickstart/cpp.md), mlpack also
provides [command-line programs](quickstart/cli.md), and bindings to the
[Python](quickstart/python.md), [R](quickstart/R.md),
[Julia](quickstart/julia.md), and [Go](quickstart/go.md) languages.

_If you use mlpack, please [cite the software](citation.md)._

## mlpack basics

Installing mlpack can be done using the
[instructions in the README](README.md#3-installing-and-using-mlpack-in-c);
or the [Windows build guide](user/build_windows.md).
The following basic guides are *highly recommended* before using mlpack.

 * ***First steps***:
   - [mlpack C++ quickstart](quickstart/cpp.md): create a couple simple C++
     programs that use mlpack
   - [Sample Windows mlpack C++ application](user/sample_ml_app.md): create a
     working mlpack Windows program using Visual Studio

 * ***Basics of matrices and data in mlpack***:
   - [Matrices and data in mlpack](user/matrices.md)
   - [Loading and saving mlpack objects](user/load_save.md)

 * ***Reference for mlpack core classes***:
   - [Core mlpack documentation](user/core.md)

 * ***Using mlpack natively with our extensions in Python, R, CLI, Julia, and Go***:
   - [Links to quickstarts and references](#bindings-to-other-languages)

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

### Classification algorithms

Classify points as discrete labels (`0`, `1`, `2`, ...).

 * [`AdaBoost`](user/methods/adaboost.md): Adaptive Boosting
 * [`DecisionTree`](user/methods/decision_tree.md): ID3-style decision tree
   classifier
 * [`HoeffdingTree`](user/methods/hoeffding_tree.md): streaming/incremental
   decision tree classifier
 * [`LinearSVM`](user/methods/linear_svm.md): simple linear support vector
   machine classifier
 * [`LogisticRegression`](user/methods/logistic_regression.md): L2-regularized
   logistic regression (two-class only)
 * [`NaiveBayesClassifier`](user/methods/naive_bayes_classifier.md): simple
   multi-class naive Bayes classifier
 * [`Perceptron`](user/methods/perceptron.md): simple Perceptron classifier
 * [`RandomForest`](user/methods/random_forest.md): parallelized random forest
   classifier
 * [`SoftmaxRegression`](user/methods/softmax_regression.md): L2-regularized
   softmax regression (i.e. multi-class logistic regression)

### Regression algorithms

Predict continuous values.

 * [`BayesianLinearRegression`](user/methods/bayesian_linear_regression.md):
   Bayesian L2-penalized linear regression
 * [`DecisionTreeRegressor`](user/methods/decision_tree_regressor.md): ID3-style
   decision tree regressor
 * [`LARS`](user/methods/lars.md): Least Angle Regression (LARS), L1-regularized
   and L2-regularized
 * [`LinearRegression`](user/methods/linear_regression.md): L2-regularized
   linear regression (ridge regression)

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

Tools for assembling a full data science pipeline.

 * [Cross-validation](user/cv.md): k-fold cross-validation tools for any mlpack
   algorithm
 * [Hyperparameter tuning](user/hpt.md): generic hyperparameter tuner to find
   good hyperparameters for any mlpack algorithm

## Bindings to other languages

mlpack's bindings to other languages have less complete functionality than
mlpack in C++, but almost all the same algorithms are available.

| ***Python*** | -- | [quickstart](quickstart/python.md) | -- | [reference](https://www.mlpack.org/doc/python_documentation.html) |
| ***Julia*** | -- | [quickstart](quickstart/julia.md) | -- | [reference](https://www.mlpack.org/doc/julia_documentation.html) |
| ***R*** | -- | [quickstart](quickstart/R.md) | -- | [reference](https://www.mlpack.org/doc/r_documentation.html)
| ***Command-line programs*** | -- | [quickstart](quickstart/cli.md) | -- | [reference](https://www.mlpack.org/doc/cli_documentation.html) |
| ***Go*** | -- | [quickstart](quickstart/go.md) | -- | [reference](https://www.mlpack.org/doc/go_documentation.html) |

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
