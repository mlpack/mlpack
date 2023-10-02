## `DecisionTree`

The `DecisionTree` class implements a decision tree classifier that supports
numerical and categorical features, by default using Gini gain to choose which
feature to split on.  The class offers several template parameters and several
constructor parameters that can be used to control the behavior of the tree.

#### Basic usage example excerpt:

```c++
DecisionTree tree(3);                       // Step 1: construct object.
tree.Train(data, labels, 3);                // Step 2: train model.
tree.Classify(test_data, test_predictions); // Step 3: use model to classify.
```

#### Quick links:

 * [Constructors](#constructors): create `DecisionTree` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom behavior.

#### See also:

 * [`DecisionTreeRegressor`](#decision_tree_regressor) <!-- TODO: fix link! -->
 * [Random forests](#random_forests) <!-- TODO: fix link! -->
 * [mlpack classifiers](#mlpack_classifiers) <!-- TODO: fix link! -->
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Decision tree learning on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

### Constructors

Construct a `DecisionTree` object using one of the constructors below.  Defaults
and types are detailed in the [Constructor Parameters](#constructor-parameters)
section below.

#### Forms:

 * `DecisionTree()`
 * `DecisionTree(numClasses)`
   - Initialize tree without training.
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Classify()`](#classification).

---

 * `DecisionTree(data, labels, numClasses)`
 * `DecisionTree(data, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical-only data.
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).

---

 * `DecisionTree(data, datasetInfo, labels, numClasses)`
 * `DecisionTree(data, datasetInfo, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed categorical data.
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).

---

 * `DecisionTree(data, datasetInfo, labels, numClasses, weights)`
 * `DecisionTree(data, datasetInfo, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted mixed categorical data.
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).
   - `weights` should be a vector of length `data.n_cols`, containing instance
     weights for each point in `data`.

---

<!-- TODO: weighted numerical-only constructors -->

#### Constructor Parameters:

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
| `labels` | [`arma::Row<size_t>`]('../matrices.md') | Training labels, between `0` and `numClasses - 1` (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`]('../matrices.md') | Weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `minimumLeafSize` | `size_t` | Minimum number of points in each leaf node. | `10` |
| `minimumGainSplit` | `double` | Minimum gain for a node to split. | `1e-7` |
| `maximumDepth` | `size_t` | Maximum depth for the tree. (0 means no limit.) | `0` |

***Note:*** different types can be used for `data` and `weights` (e.g.,
`arma::fmat`, `arma::sp_mat`).  However, the element type of `data` and
`weights` must match; for example, if `data` has type `arma::fmat`, then
`weights` must have type `arma::frowvec`.

### Training

If training is not done as part of the constructor call, it can be done with one
of the versions of the `Train()` member function.  For an instance of
`DecisionTree` named `tree`, the following functions for training are available:

 * `tree.Train(data, labels, numClasses)`
 * `tree.Train(data, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical-only data.
   - If hyperparameters are not specified, default values are used.

---

 * `tree.Train(data, labels, numClasses, weights)`
 * `tree.Train(data, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted numerical-only data.
   - If hyperparameters are not specified, default values are used.

---

 * `tree.Train(data, datasetInfo, labels, numClasses)`
 * `tree.Train(data, datasetInfo, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed categorical data.
   - If hyperparameters are not specified, default values are used.

---

 * `tree.Train(data, datasetInfo, labels, numClasses, weights)`
 * `tree.Train(data, datasetInfo, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted mixed categorical data.
   - If hyperparameters are not specified, default values are used.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Note***: training is not incremental.  A second call to `Train()` will
retrain the decision tree from scratch.

### Classification

Once a `DecisionTree` is trained, the `Classify()` member function can be used
to make class predictions for new data.  Defaults and types are detailed in the
[Classification Parameters](#classification-parameters) section below.


#### Forms:

 * `size_t predictedClass = tree.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `tree.Classify(point, prediction, probabilities_vec)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.
    - The class probabilities are stored in `probabilities_vec`, which is set to
      length `num_classes`.
    - The probability of class `i` can be accessed with `probabilities_vec[i]`.

---

 * `tree.Classify(data, predictions)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The predicted classes of each point is stored in `predictions`, which is
      set to length `data.n_cols`.
    - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `tree.Classify(data, predictions, probabilities)`
    - ***(Multi-point)***
    - Classify a set of points and compute class probabilities for each point.
    - The predicted classes of each point is stored in `predictions`, which is
      set to length `data.n_cols`.
    - The prediction for data point `i` can be accessed with `predictions[i]`.
    - The class probabilities for each point are stored in `probabilities`,
      which is set to size `num_classes` by `data.n_cols`.
    - The probability of class `j` for data point `i` can be accessed with
      `probabilities(j, i)`.

---

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
| _single-point_ | `probabilities_vec` | [`arma::vec&`](../matrices.md) | `arma::vec&` to store class probabilities into. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into. |
| _multi-point_ | `probabilities` | [`arma::mat&`](../matrices.md) | Matrix to store class probabilities into (number of rows will be equal to number of classes). |

***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other functionality

<!-- TODO: we should point directly to the documentation of those functions -->

 * A `DecisionTree` can be serialized with [`data::Save()`](../formats.md) and
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

### Simple examples

Train a decision tree on random numeric data and predict labels on a test set:

```c++
// 1000 random points in 10 dimensions.
arma::mat dataset(10, 1000);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
DecisionTree<> tree(data, labels, 5);

// Create test data (500 points).
arma::mat testDataset(10, 500);
arma::Row<size_t> predictions;
tree.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```

---

Train a decision tree on random mixed categorical data:

```c++
// Load a categorical dataset.
arma::mat dataset;
data::DatasetInfo info;
// See https://datasets.mlpack.org/iris.arff.
data::Load("iris.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/iris_labels.csv.
data::Load("iris_labels.csv", labels, true);

// Create the tree.
DecisionTree<> tree;
// Train on the given dataset, specifying a minimum leaf size of 1.
tree.Train(dataset, info, labels, 3 /* classes */, 1 /* minimum leaf size */);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/iris_test.arff.
data::Load("iris_test.arff", testDataset, info, true);

// Predict class of first test point.
const size_t firstPrediction = tree.Classify(testDataset.col(0));
std::cout << "Predicted class of first test point is " << firstPrediction << "."
    << std::endl;

// Predict class and probabilities of second test point.
size_t secondPrediction;
arma::vec secondProbabilities;
tree.Classify(testDataset.col(1), secondPrediction, secondProbabilities);
std::cout << "Class probabilities of second test point: " <<
    secondProbabilities.t();
```

---

Load a tree and print some information about it.

```c++
DecisionTree tree;
// This call assumes a tree called "tree" has already been saved to `tree.bin`
// with data::Save().
data::Load("tree.bin", "tree", tree, true);

if (tree.NumChildren() > 0)
{
  std::cout << "The split dimension of the root node of the tree in `tree.bin` "
      << "is dimension " << tree.SplitDimension() << "." << std::endl;
}
else
{
  std::cout << "The tree in `tree.bin` is a leaf (it has no children)."
      << std::endl;
}
```

---

See also the following fully-working examples:

 - [Loan default prediction with `DecisionTree`](https://github.com/mlpack/examples/blob/master/loan_default_prediction_with_decision_tree/loan-default-prediction-with-decision-tree-cpp.ipynb)

### Advanced Functionality: Template Parameters

#### Using different element types.

`DecisionTree`'s constructors, `Train()`, and `Classify()` functions support
any data type, so long as it supports the Armadillo matrix API.  So, learning
can be done on single-precision floating-point data:

```c++
// 1000 random points in 10 dimensions.
arma::fmat dataset(10, 1000);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
DecisionTree<> tree(data, labels, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500);
arma::Row<size_t> predictions;
tree.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```

---

#### Fully custom behavior.

The `DecisionTree<>` class also supports several template parameters, which can
be used for custom behavior during learning.  The full signature of the class is
as follows:

```c++
DecisionTree<FitnessFunction,
             NumericSplitType,
             CategoricalSplitType,
             DimensionSelectionType,
             NoRecursion>
```

 * `FitnessFunction`: the measure of goodness to use when deciding on tree
   splits
 * `NumericSplitType`: the strategy used for finding splits on numeric data
   dimensions
 * `CategoricalSplitType`: the strategy used for finding splits on categorical
   data dimensions
 * `DimensionSelectionType`: the strategy used for proposing dimensions to
   attempt to split on
 * `NoRecursion`: a boolean indicating whether or not to build a tree or a stump
   (one level tree)

Below, details are given for the requirements of each of these template types.

---

#### `FitnessFunction`

 * Specifies the fitness function to use when learning a decision tree.
 * The `GiniGain` _(default)_ and `InformationGain` classes are available for
   drop-in usage.
 * A custom class must implement three functions:

```c++
// You can use this as a starting point for implementation.
class CustomFitnessFunction
{
  // Return the range (difference between maximum and minimum gain values).
  double Range(const size_t numClasses);

  // Compute the gain for the given vector of labels, where `labels[i]` has an
  // associated instance weight `weights[i]`.
  //
  // `RowType` and `WeightVecType` will be vector types following the Armadillo
  // API.  If `UseWeights` is `false`, then the `weights` vector should be
  // ignored (e.g. the labels are not weighted).
  template<bool UseWeights, typename RowType, typename WeightVecType>
  double Evaluate(const RowType& labels,
                  const size_t numClasses,
                  const WeightVecType& weights);

  // Compute the gain for the given counted set of labels, where `counts[i]`
  // contains the number of points with label `i`.  There are `totalCount`
  // labels total, and `counts` has length `numClasses`.
  //
  // `UseWeights` is ignored, and `CountType` will be an integral type (e.g.
  // `size_t`).
  template<bool UseWeights, typename CountType>
  double EvaluatePtr(const CountType* counts,
                     const size_t numClasses,
                     const CountType totalCount);
};
```

---

 * `NumericSplitType`
    - Specifies the strategy to be used during training when splitting a numeric
      feature.
    - The `BestBinaryNumericSplit` _(default)_ class is available for drop-in
      usage and finds the best binary (two-way) split among all possible binary
      splits.
    - The `RandomBinaryNumericSplit` class is available for drop-in usage and
      will select a split randomly between the minimum and maximum values of a
      dimension.  It is very efficient but does not yield splits that maximize
      the gain.  (Used by the `ExtraTrees` variant of
      [`RandomForest`](#random_forest).) <!-- TODO: fix link! -->
    - A custom class must implement three functions and have an internal
      structure `AuxiliarySplitInfo` that is used at classification time.

```c++
class CustomNumericSplit
{
 public:
  // If a split with better resulting gain than `bestGain` is found, then
  // information about the new, better split should be stored in `splitInfo` and
  // `aux`.  Specifically, a split is better than `bestGain` if the sum of the
  // gains that the children will have (call this `sumChildrenGains`) is
  // sufficiently better than the gain of the unsplit node (call this
  // `unsplitGain`):
  //
  //    split if `sumChildrenGains - unsplitGain > bestGain`, and
  //             `sumChildrenGains - unsplitGain > minimumGainSplit`, and
  //             each child will have at least `minimumLeafSize` points
  //
  // The new best split value should be returned (or anything greater than or
  // equal to `bestGain` if no better split is found).
  //
  // If a new best split is found, then `splitInfo` and `aux` should be
  // populated with the information that will be needed for
  // `CalculateDirection()` to successfully choose the child for a given point.
  // `splitInfo` should be set to a vector of length 1.  The format of `aux` is
  // arbitrary and is detailed more below.
  //
  // If `UseWeights` is false, the vector `weights` should be ignored.
  // Otherwise, they are instance weighs for each value in `data` (one dimension
  // of the input data).
  template<bool UseWeights, typename VecType, typename WeightVecType>
  double SplitIfBetter(const double bestGain,
                       const VecType& data,
                       const arma::Row<size_t>& labels,
                       const size_t numClasses,
                       const WeightVecType& weights,
                       const size_t minimumLeafSize,
                       const double minimumGainSplit,
                       arma::vec& splitInfo,
                       AuxiliarySplitInfo& aux);

  // Return the number of children for a given split (stored as the single
  // element from `splitInfo` and auxiliary data `aux` in `SplitIfBetter()`).
  size_t NumChildren(const double& splitInfo,
                     const AuxiliarySplitInfo& aux);

  // Given a point with value `point`, and split information `splitInfo` and
  // `aux`, return the index of the child that corresponds to the point.  So,
  // e.g., if the split type was a binary split on the value `splitInfo`, you
  // might return `0` if `point < splitInfo`, and `1` otherwise.
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const double& splitInfo,
      const AuxiliarySplitInfo& /* aux */);

  // This class can hold any extra data that is necessary to encode a split.  It
  // should only be non-empty if a single `double` value cannot be used to hold
  // the information corresponding to a split.
  class AuxiliarySplitInfo { };
};
```

---

#### `CategoricalSplitType`

 * Specifies the strategy to be used during training when splitting a
    categorical feature.
 * The `AllCategoricalSplit` _(default)_ is available for drop-in usage and
   splits all categories into their own node.
 * A custom class must implement three functions and have an internal
   structure `AuxiliarySplitInfo` that is used at classification time:

```c++
class CustomCategoricalSplit
{
 public:
  // If a split with better resulting gain than `bestGain` is found, then
  // information about the new, better split should be stored in `splitInfo` and
  // `aux`.  Specifically, a split is better than `bestGain` if the sum of the
  // gains that the children will have (call this `sumChildrenGains`) is
  // sufficiently better than the gain of the unsplit node (call this
  // `unsplitGain`):
  //
  //    split if `sumChildrenGains - unsplitGain > bestGain`, and
  //             `sumChildrenGains - unsplitGain > minimumGainSplit`, and
  //             each child will have at least `minimumLeafSize` points
  //
  // The new best split value should be returned (or anything greater than or
  // equal to `bestGain` if no better split is found).
  //
  // If a new best split is found, then `splitInfo` and `aux` should be
  // populated with the information that will be needed for
  // `CalculateDirection()` to successfully choose the child for a given point.
  // `splitInfo` should be set to a vector of length 1.  The format of `aux` is
  // arbitrary and is detailed more below.
  //
  // If `UseWeights` is false, the vector `weights` should be ignored.
  // Otherwise, they are instance weighs for each value in `data` (one
  // categorical dimension of the input data, which takes values between `0` and
  // `numCategories - 1`).
  template<bool UseWeights, typename VecType, typename LabelsType,
           typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const size_t numCategories,
      const LabelsType& labels,
      const size_t numClasses,
      const WeightVecType& weights,
      const size_t minimumLeafSize,
      const double minimumGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux);

  // Return the number of children for a given split (stored as the single
  // element from `splitInfo` and auxiliary data `aux` in `SplitIfBetter()`).
  size_t NumChildren(const double& splitInfo,
                     const AuxiliarySplitInfo& aux);

  // Given a point with (categorical) value `point`, and split information
  // `splitInfo` and `aux`, return the index of the child that corresponds to
  // the point.  So, e.g., for `AllCategoricalSplit`, which splits a categorical
  // dimension into one child for each category, this simply returns `point`.
  template<typename ElemType>
  static size_t CalculateDirection(
      const ElemType& point,
      const double& splitInfo,
      const AuxiliarySplitInfo& /* aux */);

  // This class can hold any extra data that is necessary to encode a split.  It
  // should only be non-empty if a single `double` value cannot be used to hold
  // the information corresponding to a split.
  class AuxiliarySplitInfo { };
};
```

---

#### `DimensionSelectionType`

 * When splitting a decision tree, `DimensionSelectionType` proposes possible
   dimensions to try splitting on.
 * `AllDimensionSplit` _(default)_ is available for drop-in usage and proposes
   all dimensions for splits.
 * `MultipleRandomDimensionSelect`, constructed as
   `MultipleRandomDimensionSplit(n)`, selects `n` different random dimensions as
   candidates at each decision tree node.
 * A custom class must implement three simple functions:

```c++
class CustomDimensionSelect
{
 public:
  // Get the first dimension to try.
  // This should return a value between `0` and `data.n_rows`.
  size_t Begin();

  // Get the next dimension to try.  Note that internal state can be used to
  // track which candidate dimension is currently being looked at.
  // This should return a value between `0` and `data.n_rows`.
  size_t Next();

  // Get a value indicating that all dimensions have been tried.
  size_t End() const;

  // The usage pattern of `DimensionSelectionType` by `DecisionTree` is as
  // follows, assuming that `dim` is an instantiated `DimensionSelectionType`
  // object:
  //
  // for (size_t dim = dim.Begin(); dim != dim.End(); dim = dim.Next())
  // {
  //   // ... try to split on dimension `dim` ...
  // }
};
```

---

#### `NoRecursion`

 * A `bool` value that indicates whether a decision tree should be
   constructed recursively.
 * If `true` _(default)_, a full decision tree will be built.
 * If `false`, only the root node will be split (producing a decision
   stump).
