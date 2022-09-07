# Density estimation tree (DET) tutorial

DETs perform the unsupervised task of density estimation using decision trees.
Using a trained density estimation tree (DET), the density at any particular
point can be estimated very quickly (`O(log n)` time, where `n` is the number of
points the tree is built on).

The details of this work is presented in the following paper:

```
@inproceedings{ram2011density,
  title={Density estimation trees},
  author={Ram, P. and Gray, A.G.},
  booktitle={Proceedings of the 17th ACM SIGKDD International Conference on
      Knowledge Discovery and Data Mining},
  pages={627--635},
  year={2011},
  organization={ACM}
}
```

mlpack provides:

 - a simple command-line executable to perform density estimation and related
   analyses using DETs
 - a generic C++ class (`DTree`) which provides various functionality for the
   DETs
 - a set of functions in the namespace `mlpack::det` to perform cross-validation
   for the task of density estimation with DETs

## Command-line `mlpack_det`

*(Note: this section was written for the command-line program `mlpack_det`, but
a `det()` function is available for other languages via mlpack's bindings
system.  The options are so similar that it is easy to adapt the examples here
to another language.)*

The command line arguments of this program can be viewed using the `-h` option:

```sh
$ mlpack_det -h
Density Estimation With Density Estimation Trees

  This program performs a number of functions related to Density Estimation
  Trees.  The optimal Density Estimation Tree (DET) can be trained on a set of
  data (specified by --training_file or -t) using cross-validation (with number
  of folds specified by --folds).  This trained density estimation tree may then
  be saved to a model file with the --output_model_file (-M) option.

  The variable importances of each dimension may be saved with the --vi_file
  (-i) option, and the density estimates on each training point may be saved to
  the file specified with the --training_set_estimates_file (-e) option.

  This program also can provide density estimates for a set of test points,
  specified in the --test_file (-T) file.  The density estimation tree used for
  this task will be the tree that was trained on the given training points, or a
  tree stored in the file given with the --input_model_file (-m) parameter.  The
  density estimates for the test points may be saved into the file specified
  with the --test_set_estimates_file (-E) option.


Options:

  --folds (-f) [int]            The number of folds of cross-validation to
                                perform for the estimation (0 is LOOCV)  Default
                                value 10.
  --help (-h)                   Default help info.
  --info [string]               Get help on a specific module or option.
                                Default value ''.
  --input_model_file (-m) [string]
                                File containing already trained density
                                estimation tree.  Default value ''.
  --max_leaf_size (-L) [int]    The maximum size of a leaf in the unpruned,
                                fully grown DET.  Default value 10.
  --min_leaf_size (-l) [int]    The minimum size of a leaf in the unpruned,
                                fully grown DET.  Default value 5.
  --output_model_file (-M) [string]
                                File to save trained density estimation tree to.
                                 Default value ''.
  --test_file (-T) [string]     A set of test points to estimate the density of.
                                 Default value ''.
  --test_set_estimates_file (-E) [string]
                                The file in which to output the estimates on the
                                test set from the final optimally pruned tree.
                                Default value ''.
  --training_file (-t) [string]
                                The data set on which to build a density
                                estimation tree.  Default value ''.
  --training_set_estimates_file (-e) [string]
                                The file in which to output the density
                                estimates on the training set from the final
                                optimally pruned tree.  Default value ''.
  --verbose (-v)                Display informational messages and the full list
                                of parameters and timers at the end of
                                execution.
  --version (-V)                Display the version of mlpack.
  --vi_file (-i) [string]       The file to output the variable importance
                                values for each feature.  Default value ''.

For further information, including relevant papers, citations, and theory,
consult the documentation found at http://www.mlpack.org or included with your
distribution of mlpack.
```

### Plain-vanilla density estimation

We can just train a DET on the provided data set `S`.  Like all datasets
mlpack uses, the data should be row-major (mlpack transposes data when it
is loaded; internally, the data is column-major---see [this
page](../user/matrices.md) for more information).

```sh
$ mlpack_det -t dataset.csv -v
```

By default, `mlpack_det` performs 10-fold cross-validation (using the
alpha-pruning regularization for decision trees). To perform LOOCV
(leave-one-out cross-validation), which can provide better results but will take
longer, use the following command:

```sh
$ mlpack_det -t dataset.csv -f 0 -v
```

To perform `k`-fold crossvalidation, use `-f k` (or `--folds k`). There are
certain other options available for training. For example, in the construction
of the initial tree, you can specify the maximum and minimum leaf sizes. By
default, they are 10 and 5 respectively; you can set them using the `-M`
(`--max_leaf_size`) and the `-N` (`--min_leaf_size`) options.

```sh
$ mlpack_det -t dataset.csv -M 20 -N 10
```

In case you want to output the density estimates at the points in the training
set, use the `-e` (`--training_set_estimates_file`) option to specify the output
file to which the estimates will be saved.  The first line in
`density_estimates.txt` will correspond to the density at the first point in the
training set.  Note that the logarithm of the density estimates are given, which
allows smaller estimates to be saved.

```sh
$ mlpack_det -t dataset.csv -e density_estimates.txt -v
```

### Estimation on a test set

Often, it is useful to train a density estimation tree on a training set and
then obtain density estimates from the learned estimator for a separate set of
test points.  The `-T` (`--test_file`) option allows specification of a set of
test points, and the `-E` (`--test_set_estimates_file`) option allows
specification of the file into which the test set estimates are saved.  Note
that the logarithm of the density estimates are saved; this allows smaller
values to be saved.

```sh
$ mlpack_det -t dataset.csv -T test_points.csv -E test_density_estimates.txt -v
```

### Computing the variable importance

The variable importance (with respect to density estimation) of the different
features in the data set can be obtained by using the `-i` (`--vi_file`) option.
This outputs the absolute (as opposed to relative) variable importance of the
all the features into the specified file.

```sh
$ mlpack_det -t dataset.csv -i variable_importance.txt -v
```

### Saving trained DETs

The `mlpack_det` program is capable of saving a trained DET to a file for later
usage.  The `--output_model_file` or `-M` option allows specification of the
file to save to.  In the example below, a DET trained on `dataset.csv` is saved
to the file `det.xml`.

```sh
$ mlpack_det -t dataset.csv -M det.xml -v
```

### Loading trained DETs

A saved DET can be used to perform any of the functionality in the examples
above.  A saved DET is loaded with the `--input_model_file` or `-m` option.  The
example below loads a saved DET from `det.xml` and outputs density estimates on
the dataset `test_dataset.csv` into the file `estimates.csv`.

```sh
$ mlpack_det -m det.xml -T test_dataset.csv -E estimates.csv -v
```

## The `DTree` class

This class implements density estimation trees.  Below is a simple example which
initializes a density estimation tree.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The dataset matrix, on which to learn the density estimation tree.
extern arma::Mat<float> data;

// Initialize the tree.  This function also creates and saves the bounding box
// of the data.  Note that it does not actually build the tree.
DTree<> det(data);
```

### Public functions

The function `Grow()` greedily grows the tree, adding new points to the tree.
Note that the points in the dataset will be reordered.  This should only be run
on a tree which has not already been built.  In general, it is more useful to
use the `Trainer()` function, detailed later.

```c++
// This keeps track of the data during the shuffle that occurs while growing the
// tree.
arma::Col<size_t> oldFromNew(data.n_cols);
for (size_t i = 0; i < data.n_cols; i++)
  oldFromNew[i] = i;

// This function grows the tree down to the leaves. It returns the current
// minimum value of the regularization parameter alpha.
size_t maxLeafSize = 10;
size_t minLeafSize = 5;

double alpha = det.Grow(data, oldFromNew, false, maxLeafSize, minLeafSize);
```

Note that the alternate volume regularization should not be used (see
[#238](https://github.com/mlpack/mlpack/issues/238)).

To estimate the density at a given query point, use the following code.  Note
that the logarithm of the density is returned.

```c++
// For a given query, you can obtain the density estimate.
extern arma::Col<float> query;
extern DTree* det;
double estimate = det->ComputeValue(&query);
```

Computing the *variable importance* of each feature for the given DET.

```c++
// The data matrix and density estimation tree.
extern arma::mat data;
extern DTree* det;

// The variable importances will be saved into this vector.
arma::Col<double> varImps;

// You can obtain the variable importance from the current tree.
det->ComputeVariableImportance(varImps);
```

## The `mlpack::det` namespace

The functions in this namespace allows the user to perform tasks with the
`DTree` class.  Most importantly, the `Trainer()` method allows the full
training of a density estimation tree with cross-validation.  There are also
utility functions which allow printing of leaf membership and variable
importance.

### Utility functions

The code below details how to train a density estimation tree with
cross-validation.

```c++
#include <mlpack.hpp>

using namespace mlpack;

// The dataset matrix, on which to learn the density estimation tree.
extern arma::Mat<float> data;

// The number of folds for cross-validation.
const size_t folds = 10; // Set folds = 0 for LOOCV.

const size_t maxLeafSize = 10;
const size_t minLeafSize = 5;

// Train the density estimation tree with cross-validation.
DTree<>* dtree_opt = Trainer(data, folds, false, maxLeafSize, minLeafSize);
```

Note that the alternate volume regularization should be set to false because it
has known bugs (see [#238](https://github.com/mlpack/mlpack/issues/238))..

To print the class membership of leaves in the tree into a file, see the
following code.

```c++
extern arma::Mat<size_t> labels;
extern DTree* det;
const size_t numClasses = 3; // The number of classes must be known.

extern string leafClassMembershipFile;

PrintLeafMembership(det, data, labels, numClasses, leafClassMembershipFile);
```

Note that you can find the number of classes with `max(labels) + 1`.  The
variable importance can also be printed to a file in a similar manner.

```c++
extern DTree* det;

extern string variableImportanceFile;
const size_t numFeatures = data.n_rows;

PrintVariableImportance(det, numFeatures, variableImportanceFile);
```

## Further documentation

For further documentation on the `DTree` class, consult the comments in the
source code, in `mlpack/methods/det/`.
