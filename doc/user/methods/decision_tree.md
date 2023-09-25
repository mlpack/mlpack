## `DecisionTree`

The `DecisionTree` class implements a decision tree classifier that supports
numerical and categorical features, by default using Gini gain to choose which
feature to split on.  The class offers several template parameters and several
constructor parameters that can be used to control the behavior of the tree.



### Constructors

*Forms*:

| **signature** | **description** |
|---------------|-----------------|
| `DecisionTree(numClasses)` | Initialize tree without training. |
| `DecisionTree(data, labels, numClasses)` | Train on numerical-only data. |
| `DecisionTree(data, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)` | Train on numerical-only data with hyperparameters. |
| `DecisionTree(data, datasetInfo, labels, numClasses)` | Train on mixed categorical data. |
| `DecisionTree(data, datasetInfo, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth) | Train on mixed categorical data with hyperparameters. |
| `DecisionTree(data, datasetInfo, labels, numClasses, weights)` | Train on weighted mixed categorical data. |
| `DecisionTree(data, datasetInfo, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)` | Train on weighted mixed categorical data with hyperparameters. |

<!-- TODO: weighted numerical-only constructors -->

*Parameters*:

<!-- TODOs for table below:
    * better link for column-major matrices
    * better link for working with categorical data in straightforward terms
    * update matrices.md to include a section on labels and NormalizeLabels()
    * add a bit about instance weights in matrices.md
 -->
| **type** | **name** | **description** | **default** |
|----------|----------|-----------------|-------------|
| [`arma::mat`](../matrices.md) | **`data`** | [Column-major](../matrices.md) training matrix. | _(N/A)_ |
| [`data::DatasetInfo`](../../tutorials/datasetmapper.md) | **`datasetInfo`** | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| [`arma::Row<size_t>`]('../matrices.md') | **`labels`** | Training labels, between `0` and `numClasses - 1` (inclusive). | _(N/A)_ |
| [`arma::rowvec`]('../matrices.md') | **`weights`** | Weights for each training point. | _(N/A)_ |
| `size_t` | **`numClasses`** | Number of classes in the dataset. | _(N/A)_ |
| `size_t` | **`minimumLeafSize`** | Minimum number of points in each leaf node. | `10` |
| `double` | **`minimumGainSplit`** | Minimum gain for a node to split. | `1e-7` |
| `size_t` | **`maximumDepth`** | Maximum depth for the tree. (0 means no limit.) | `0` |

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` member function, which has several overloads.

 * `tree.Train(`_`data`_`, `_`labels`_`, `_`numClasses`_`)` (train on numerical-only data)
 * `tree.Train(`_`data`_`, `_`labels`_`, `_`numClasses`_`, `_`minimumLeafSize`_`, `_`minimumGainSplit`_`, `_`maximumDepth`_`)` (train on numerical-only data with hyperparameters)

 * `tree.Train(`_`data`_`, `_`labels`_`, `_`numClasses`_`, `_`weights`_`)` (train on weighted numerical-only data)
 * `tree.Train(`_`data`_`, `_`labels`_`, `_`numClasses`_`, `_`weights`_`, `_`minimumLeafSize`_`, `_`minimumGainSplit`_`, `_`maximumDepth`_`)` (train on weighted numerical-only data with hyperparameters)

 * `tree.Train(`_`data`_`, `_`datasetInfo`_`, `_`labels`_`, `_`numClasses`_`)` (train on mixed categorical data)
 * `tree.Train(`_`data`_`, `_`datasetInfo`_`, `_`labels`_`, `_`numClasses`_`, `_`minimumLeafSize`_`, `_`minimumGainSplit`_`, `_`maximumDepth`_`)` (train on mixed categorical data with hyperparameters)

 * `tree.Train(`_`data`_`, `_`datasetInfo`_`, `_`labels`_`, `_`numClasses`_`, `_`weights`_`)` (train on weighted mixed categorical data)
 * `tree.Train(`_`data`_`, `_`datasetInfo`_`, `_`labels`_`, `_`numClasses`_`,
   `_`weights`_`, `_`minimumLeafSize`_`, `_`minimumGainSplit`_`, `_`maximumDepth`_`)` (train on weighted mixed categorical data with hyperparameters)

Types of each argument are the same as in the table for constructors above.

### Classification

Once a `DecisionTree` is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t class = tree.Classify(`_`point`_`)` (classify a single point)
 * `tree.Classify(`_`point`_`, `_`prediction`_`, `_`probabilities`_`)` (classify a single point and compute class probabilities)

| **type** | **name** | **description** |
|----------|----------|-----------------|
| [`arma::vec`](../matrices.md) | **`point`** | Single point for classification. |
| `size_t&` | **`prediction`** | `size_t` to store class prediction into. |
| [`arma::vec&`](../matrices.md) | **`probabilities`** | `arma::vec&` to store class probabilities into. |

 * `tree.Classify(`_`data`_`, `_`predictions`_`)` (classify a set of points)
 * `tree.Classify(`_`data`_`, `_`predictions`_`, `_`probabilities`_`)` (classify a set of points and compute class probabilities for each point)

| **type** | **name** | **description** |
|----------|----------|-----------------|
| [`arma::mat`](../matrices.md) | **`data`** | Set of [column-major](../matrices.md) points for classification. |
| [`arma::Row<size_t>&`](../matrices.md) | **`predictions`** | Vector of `size_t`s to store class prediction into. |
| [`arma::mat&`](../matrices.md) | **`probabilities`** | Matrix to store class probabilities into (number of rows will be equal to number of classes). |

### Simple examples

Train a decision tree on random numeric data:

```c++
// 1000 random points in 10 dimensions.
arma::mat dataset(1000, 10);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

DecisionTree<> tree(data, labels, 5);
```

Train a decision tree on random mixed categorical data:

```c++
categorical example -- TODO
```

<!-- TODO: link to relevant examples in the examples repository -->

### Other functionality

<!-- TODO: we should point directly to the documentation of those functions -->
 * A `DecisionTree` can be serialized with [`data::Save()`](../formats.md) an
   [`data::Load()`](../formats.md).

 * `tree.NumChildren()` will return a `size_t` indicating the number of children
   in the node `tree`.

 * `tree.Child(i)` will return a `DecisionTree` object representing the `i`th
   child of the node `tree`.

 * `tree.SplitDimension()` returns a `size_t` indicating which dimension the
   node `tree` splits on.

 * `tree.NumClasses()` returns a `size_t` indicating the number of classes the
   tree was trained on.

For complete functionality, the [source
code](/src/mlpack/methods/decision_tree/decision_tree.hpp) can be consulted.
Each method is fully documented.

### Advanced Functionality: Template Parameters

The `DecisionTree<>` class also supports several template parameters.

<!-- TODO: this section -->

### See Also
