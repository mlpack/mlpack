## `DecisionTree`

The `DecisionTree` class implements a decision tree classifier that supports
numerical and categorical features, by default using Gini gain to choose which
feature to split on.  The class offers several template parameters and several
constructor parameters that can be used to control the behavior of the tree.

*Example excerpt*:

```c++
DecisionTree tree(3); // [Step 1](#constructors): construct object.
tree.Train(data, labels, 3); // [Step 2](#training): train model.
tree.Classify(test_data, test_predictions); // [Step 3](#classification): use model to classify points.
```



### Constructors

Construct a `DecisionTree` object using one of the constructors below.

*Forms*:

 * `DecisionTree(numClasses)`
   - Initialize tree without training.
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Classify()`](#classify).


 * `DecisionTree(data, labels, numClasses)`
 * `DecisionTree(data, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical-only data.
   - If hyperparameters are not specified, default values are used.


 * `DecisionTree(data, datasetInfo, labels, numClasses)`
 * `DecisionTree(data, datasetInfo, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed categorical data.
   - If hyperparameters are not specified, default values are used.


<!-- TODO: weighted numerical-only constructors -->

 * `DecisionTree(data, datasetInfo, labels, numClasses, weights)`
 * `DecisionTree(data, datasetInfo, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted mixed categorical data.
   - If hyperparameters are not specified, default values are used.

*Parameters*:

<!-- TODOs for table below:
    * better link for column-major matrices
    * better link for working with categorical data in straightforward terms
    * update matrices.md to include a section on labels and NormalizeLabels()
    * add a bit about instance weights in matrices.md
 -->
| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](../../tutorials/datasetmapper.md) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`]('../matrices.md') | Training labels, between `0` and `numClasses - 1` (inclusive). | _(N/A)_ |
| `weights` | [`arma::rowvec`]('../matrices.md') | Weights for each training point. | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `minimumLeafSize` | `size_t` | Minimum number of points in each leaf node. | `10` |
| `minimumGainSplit` | `double` | Minimum gain for a node to split. | `1e-7` |
| `maximumDepth` | `size_t` | Maximum depth for the tree. (0 means no limit.) | `0` |

### Training

If training is not done as part of the constructor call, it can be done with one
of the versions of the `Train()` member function.  For an instance of
`DecisionTree` named `tree`, the following functions for training are available:

 * `tree.Train(data, labels, numClasses)`
 * `tree.Train(data, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical-only data.
   - If hyperparameters are not specified, default values are used.

 * `tree.Train(data, labels, numClasses, weights)`
 * `tree.Train(data, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted numerical-only data.
   - If hyperparameters are not specified, default values are used.

 * `tree.Train(data, datasetInfo, labels, numClasses)`
 * `tree.Train(data, datasetInfo, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed categorical data.
   - If hyperparameters are not specified, default values are used.

 * `tree.Train(data, datasetInfo, labels, numClasses, weights)`
 * `tree.Train(data, datasetInfo, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted mixed categorical data.
   - If hyperparameters are not specified, default values are used.

Types of each argument are the same as in the table for constructors above.

TODO: what about incremental training?

### Classification

Once a `DecisionTree` is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t class = tree.Classify(point)`
    - Classify a single point, returning the predicted class.

 * `tree.Classify(point, prediction, probabilities_vec)`
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.
    - The class probabilities are stored in `probabilities`, which is set to
      length `num_classes`.
    - The probability of class `i` can be accessed with `probabilities_vec[i]`.

 * `tree.Classify(data, predictions)`
    - Classify a set of points.
    - The predicted classes of each point is stored in `predictions`, which is
      set to length `data.n_cols`.
    - The prediction for data point `i` can be accessed with `predictions[i]`.

 * `tree.Classify(data, predictions, probabilities)`
    - Classify a set of points and compute class probabilities for each point.
    - The predicted classes of each point is stored in `predictions`, which is
      set to length `data.n_cols`.
    - The prediction for data point `i` can be accessed with `predictions[i]`.
    - The class probabilities for each point are stored in `probabilities`,
      which is set to size `num_classes` by `data.n_cols`.
    - The probability of class `j` for data point `i` can be accessed with
      `probabilities(j, i)`.

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| `prediction` | `size_t&` | `size_t` to store class prediction into. |
| `probabilities_vec` | [`arma::vec&`](../matrices.md) | `arma::vec&` to store class probabilities into. |
| | | |
| `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md) points for classification. |
| `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into. |
| `probabilities` | [`arma::mat&`](../matrices.md) | Matrix to store class probabilities into (number of rows will be equal to number of classes). |

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
