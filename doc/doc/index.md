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
[Python](quickstart/python.md), [R](quickstart/r.md),
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
     * [Core math utilities](user/core/math.md)
     * [Distances](user/core/distances.md)
     * [Distributions](user/core/distributions.md)
     * [Kernels](user/core/kernels.md)

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

***NOTE:*** this documentation is still under construction and so some
algorithms that mlpack implements are not yet listed here.  For now, see
[the mlpack/methods directory](https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods)
for a full list of algorithms.

Group points into clusters.

 * [`MeanShift`](user/methods/mean_shift.md): clustering with the density-based
   mean shift algorithm

### Geometric algorithms

***NOTE:*** this documentation is still under construction and so no geometric
algorithms in mlpack are documented yet.  For now, see
[the mlpack/methods directory](https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods)
for a full list of algorithms.

Computations based on distance metrics.

<!-- TODO: add some -->

### Preprocessing utilities

Prepare data for machine learning algorithms.

 * [Normalizing labels](user/core/normalizing_labels.md): map labels to and from
   the range `[0, numClasses - 1]`.
 * [Dataset splitting](user/core/split.md): split a dataset into a
   training set and a test set.

***NOTE:*** this documentation is still under construction and so not all
preprocessing utilities in mlpack are documented yet.  See also
[the mlpack/methods/preprocess directory](https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods)
for a full list of algorithms.

### Transformations

***NOTE:*** this documentation is still under construction and so some
algorithms that mlpack implements are not yet listed here.  For now, see
[the mlpack/methods directory](https://github.com/mlpack/mlpack/tree/master/src/mlpack/methods)
for a full list of algorithms.

Transform data from one space to another.

 * [`AMF`](user/methods/amf.md): alternating matrix factorization
 * [`LocalCoordinateCoding`](user/methods/local_coordinate_coding.md): local
   coordinate coding with dictionary learning
 * [`LMNN`](user/methods/lmnn.md): large margin nearest neighbor (distance
   metric learning)
 * [`NCA`](user/methods/nca.md): neighborhood components analysis (distance
   metric learning)
 * [`NMF`](user/methods/nmf.md): non-negative matrix factorization
 * [`PCA`](user/methods/pca.md): principal components analysis
 * [`RADICAL`](user/methods/radical.md): robust, accurate, direct independent
   components analysis (ICA) algorithm
 * [`SparseCoding`](user/methods/sparse_coding.md): sparse coding with
   dictionary learning

### Modeling utilities

Tools for assembling a full data science pipeline.

 * [Cross-validation](user/cv.md): k-fold cross-validation tools for any mlpack
   algorithm
 * [Hyperparameter tuning](user/hpt.md): generic hyperparameter tuner to find
   good hyperparameters for any mlpack algorithm

## Bindings to other languages

mlpack's bindings to other languages have less complete functionality than
mlpack in C++, but almost all the same algorithms are available.

| ***Python*** | -- | [quickstart](quickstart/python.md) | -- | [reference](user/bindings/python.md) |
| ***Julia*** | -- | [quickstart](quickstart/julia.md) | -- | [reference](user/bindings/julia.md) |
| ***R*** | -- | [quickstart](quickstart/r.md) | -- | [reference](user/bindings/r.md)
| ***Command-line programs*** | -- | [quickstart](quickstart/cli.md) | -- | [reference](user/bindings/cli.md) |
| ***Go*** | -- | [quickstart](quickstart/go.md) | -- | [reference](user/bindings/go.md) |

## mlpack on embedded systems

mlpack is well suited for embedded systems due to the fact that it is written
in C++ and it is header-only with minimal dependencies. In the following, we are
adding a set of tutorials to allow you to experiment mlpack on various types of
these systems.

* [cross-compile and run k-NN on a Raspberry Pi 2 (armv7)](embedded/crosscompile_armv7.md)

## Examples and further documentation

 * [mlpack examples repository](https://github.com/mlpack/examples/): numerous
   fully-working example applications of mlpack, in C++ and other languages.
 * [mlpack models repository](https://github.com/mlpack/models/): complex models
   in C++ built with mlpack

For additional documentation beyond what is covered in all the resources above,
the source code should be consulted.  Each method is fully documented.

## Developer documentation

The following general documentation can be useful if you are interested in
contributing to mlpack:

 * [The mlpack community](developer/community.md)
 * [mlpack and Google Summer of Code](developer/gsoc.md)

Throughout the codebase, mlpack uses some common template parameter policies.
These are documented below.

 * [The `ElemType` policy](developer/elemtype.md): element types for data
 * [The `DistanceType` policy](developer/distances.md): distance metrics
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

## Changelog

For a list of changes in each version of mlpack, see the
[changelog](HISTORY.md).
