## `DecisionTree`

The `DecisionTree` class implements a decision tree classifier that supports
numerical and categorical features, by default using Gini gain to choose which
feature to split on.  The class offers several template parameters and several
runtime options that can be used to control the behavior of the tree.

Decision trees are useful for classifying points with _discrete labels_ (i.e.
`0`, `1`, `2`).  For predicting _continuous values_ (regression), see
[`DecisionTreeRegressor`](decision_tree_regressor.md).

#### Simple usage example:

```c++
// Train a decision tree on random numeric data and predict labels on test data:

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::DecisionTree tree;               // Step 1: create model.
tree.Train(dataset, labels, 5);          // Step 2: train model.
arma::Row<size_t> predictions;
tree.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `DecisionTree` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [`DecisionTreeRegressor`](decision_tree_regressor.md)
 * [Random forests](random_forest.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Decision tree learning on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

### Constructors

 * `tree = DecisionTree()`
   - Initialize tree without training.
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Classify()`](#classification).

---

 * `tree = DecisionTree(data, labels, numClasses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree = DecisionTree(data, labels, numClasses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on numerical-only data (optionally with instance weights).

---

 * `tree = DecisionTree(data, datasetInfo, labels, numClasses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree = DecisionTree(data, datasetInfo, labels, numClasses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on mixed categorical data (optionally with instance weights).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](../load_save.md#loading-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`](../matrices.md) | Weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `minLeafSize` | `size_t` | Minimum number of points in each leaf node. | `10` |
| `minGainSplit` | `double` | Minimum gain for a node to split. | `1e-7` |
| `maxDepth` | `size_t` | Maximum depth for the tree. (0 means no limit.) | `0` |

 * Setting `minLeafSize` too small (e.g. `1`) may cause the tree to overfit to
   its training data, and may create a very large tree.  However, setting it too
   large may cause the tree to be very small and underfit.
 * `minGainSplit` has similar behavior: if it is too small, the tree may
   overfit; if too large, it may underfit.

***Note:*** different types can be used for `data` and `weights` (e.g.,
`arma::fmat`, `arma::sp_mat`).  However, the element type of `data` and
`weights` must match; for example, if `data` has type `arma::fmat`, then
`weights` must have type `arma::frowvec`.

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `tree.Train(data, labels, numClasses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree.Train(data, labels, numClasses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on numerical-only data (optionally with instance weights).

---

 * `tree.Train(data, datasetInfo, labels, numClasses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree.Train(data, datasetInfo, labels, numClasses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on mixed categorical data (optionally with instance weights).

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes***:

 * Training is not incremental.  A second call to `Train()` will retrain the
   decision tree from scratch.

 * `Train()` returns a `double` with the final gain of the tree (the Gini gain,
   unless a different
   [`FitnessFunction` template parameter](#fully-custom-behavior) is specified.

### Classification

Once a `DecisionTree` is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t predictedClass = tree.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `tree.Classify(point, prediction, probabilitiesVec)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.
    - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `tree.Classify(data, predictions)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `tree.Classify(data, predictions, probabilities)`
    - ***(Multi-point)***
    - Classify a set of points and compute class probabilities for each point.
    - The prediction for data point `i` can be accessed with `predictions[i]`.
    - The probability of class `j` for data point `i` can be accessed with
      `probabilities(j, i)`.

---

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
| _single-point_ | `probabilitiesVec` | [`arma::vec&`](../matrices.md) | `arma::vec&` to store class probabilities into.  Will be set to length `numClasses`. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into.  Will be set to length `data.n_cols`. |
| _multi-point_ | `probabilities` | [`arma::mat&`](../matrices.md) | Matrix to store class probabilities into (number of rows will be equal to number of classes, number of columns will be equal to `data.n_cols`). |

***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other Functionality

 * A `DecisionTree` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `tree.NumChildren()` will return a `size_t` indicating the number of children
   in the node `tree`. If there was no split, zero is returned. 

 * `tree.Child(i)` will return a `DecisionTree` object representing the `i`th
   child of the node `tree`.

 * `tree.SplitDimension()` returns a `size_t` indicating which dimension the
   node `tree` splits on.

 * `tree.NumClasses()` returns a `size_t` indicating the number of classes the
   tree was trained on.

For complete functionality, the [source
code](/src/mlpack/methods/decision_tree/decision_tree.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`DecisionTree`.

---

Train a decision tree on mixed categorical data and save it:

```c++
// Load a categorical dataset.
arma::mat dataset;
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, true);

// Create the tree.
mlpack::DecisionTree tree;
// Train on the given dataset, specifying a minimum leaf size of 5.
tree.Train(dataset, info, labels, 7 /* classes */, 5 /* minimum leaf size */);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, info, true);

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

// Save the tree to `tree.bin`.
mlpack::data::Save("tree.bin", "tree", tree);
```

---

Load a tree and print some information about it.

```c++
mlpack::DecisionTree tree;
// This call assumes a tree called "tree" has already been saved to `tree.bin`
// with `data::Save()`.
mlpack::data::Load("tree.bin", "tree", tree, true);

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

 - [Loan default prediction with `DecisionTree`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/decision_tree/loan_default_prediction/loan-default-prediction-cpp.ipynb)

### Advanced Functionality: Template Parameters

#### Using different element types.

`DecisionTree`'s constructors, `Train()`, and `Classify()` functions support
any data type, so long as it supports the Armadillo matrix API.  So, for
instance, learning can be done on single-precision floating-point data:

```c++
// 1000 random points in 10 dimensions.
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
mlpack::DecisionTree tree(dataset, labels, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
tree.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```

---

#### Fully custom behavior.

The `DecisionTree` class also supports several template parameters, which can
be used for custom behavior during learning.  The full signature of the class is
as follows:

```
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
 * `NoRecursion`: a boolean indicating whether to build a tree or a stump
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

#### `NumericSplitType`

 * Specifies the strategy to be used during training when splitting a numeric
   feature.
 * The `BestBinaryNumericSplit` _(default)_ class is available for drop-in
   usage and finds the best binary (two-way) split among all possible binary
   splits.
 * The `RandomBinaryNumericSplit` class is available for drop-in usage and
   will select a split randomly between the minimum and maximum values of a
   dimension.  It is very efficient but does not yield splits that maximize
   the gain.  (Used by the `ExtraTrees` variant of
   [`RandomForest`](random_forest.md).)
 * A custom class must take a [`FitnessFunction`](#fitnessfunction) as a
   template parameter, implement three functions, and have an internal
   structure `AuxiliarySplitInfo` that is used at classification time:

```c++
template<typename FitnessFunction>
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
  //             `sumChildrenGains - unsplitGain > minGainSplit`, and
  //             each child will have at least `minLeafSize` points
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
                       const size_t minLeafSize,
                       const double minGainSplit,
                       arma::vec& splitInfo,
                       AuxiliarySplitInfo& aux);

  // Return the number of children for a given split. If there was no split, 
  // return zero. `splitInfo` and `aux` contain the split information, as set 
  // in `SplitIfBetter`. 
  size_t NumChildren(const arma::vec& splitInfo,
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
 * The `AllCategoricalSplit` _(default)_ and `BestBinaryCategoricalSplit` are~
   available for drop-in usage.
 * `AllCategoricalSplit`, the default ID3 split
   algorithm, splits all categories into their own node. This variant is simple,
   and has complexity `O(n)`, where `n` is the number of samples.
 * `BestBinaryCategoricalSplit` is the preferred algorithm of
   [the CART system](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-olshen-charles-stone).
   It will find the the best (entropy-minimizing) binary partition of the
   categories. This algorithm has complexity `O(n lg n)` in the case of binary
   outcomes, but is exponential in the number of _categories_ when there are
   more than two _classes_.~
   - ***Note***: `BestBinaryCategoricalSplit` should not be chosen when there
     are multiple classes and many categories.
 * A custom class must take a [`FitnessFunction`](#fitnessfunction) as a
   template parameter, implement three functions, and have an internal
   structure `AuxiliarySplitInfo` that is used at classification time:

```c++
template<typename FitnessFunction>
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
  //             `sumChildrenGains - unsplitGain > minGainSplit`, and
  //             each child will have at least `minLeafSize` points
  //
  // The new best split value should be returned (or anything greater than or
  // equal to `bestGain` if no better split is found).
  //
  // If a new best split is found, then `splitInfo` and `aux` should be
  // populated with the information that will be needed for
  // `CalculateDirection()` to successfully choose the child for a given point.
  // `splitInfo` should be set to a non-empty vector.  The format of `aux` is
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
      const size_t minLeafSize,
      const double minGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux);

  // Return the number of children for a given split. If there was no split,
  // return zero. `splitInfo` and `aux` contain the split information, as set
  // in `SplitIfBetter`.
  size_t NumChildren(const arma::vec& splitInfo,
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
 * `MultipleRandomDimensionSelect` proposes a different random subset of
   dimensions at each decision tree node.
    - By default each random subset is of size `sqrt(d)` where `d` is the number
      of dimensions in the data.
    - If constructed as `MultipleRandomDimensionSelect(n)` and passed to the
      constructor of `DecisionTree` or the `Train()` function, each random
      subset will be of size `n`.
 * Each `DecisionTree` [constructor](#constructors) and each version of the
   [`Train()`](#training) function optionally accept an instantiated
   `DimensionSelectionType` object as the very last parameter (after
   `maxDepth`), in case some internal state in the dimension selection mechanism
   is required.
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
 * If `true`, only the root node will be split (producing a decision
   stump).
 * If `false` _(default)_, a full decision tree will be built.
