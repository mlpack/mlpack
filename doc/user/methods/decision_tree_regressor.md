## `DecisionTreeRegressor`

The `DecisionTreeRegressor` class implements a decision tree regressor that
supports numerical and categorical features, by default using MSE (minimum
squared error) to choose which feature to split on.  The class offers several
template parameters and runtime options that can be used to control the behavior
of the tree.

The `DecisionTreeRegressor` class is useful for regressions; i.e., predicting
_continuous values_ (`0.3`, `1.2`, etc.).  For predicting _discrete labels_
(classification), see [`DecisionTree`](decision_tree.md).

#### Simple usage example:

Train a decision tree regressor on random numeric data and make predictions on a
test set:

```c++
// Train a decision tree regressor on random numeric data and make predictions.

// All data and responses are uniform random; this uses 10 dimensional data.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::rowvec responses = arma::randn<arma::rowvec>(1000);
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::DecisionTreeRegressor tree;     // Step 1: create tree.
tree.Train(dataset, responses);         // Step 2: train model.
arma::rowvec predictions;
tree.Predict(testDataset, predictions); // Step 3: use model to predict.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 0.7) << " test points predicted to have"
    << " responses greater than 0.7." << std::endl;
std::cout << arma::accu(predictions < 0) << " test points predicted to have "
    << "negative responses." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `DecisionTreeRegressor` objects.
 * [`Train()`](#training): train model.
 * [`Predict()`](#prediction): predict values with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [`DecisionTree`](decision_tree.md)
 * [Random forests](random_forest.md)
 * [mlpack regression techniques](../modeling.md#regression)
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Decision tree learning on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

### Constructors

 * `tree = DecisionTreeRegressor()`
   - Initialize tree without training.
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Predict()`](#prediction).

---

 * `tree = DecisionTreeRegressor(data, responses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree = DecisionTreeRegressor(data, responses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on numerical-only data (optionally with instance weights).

---

 * `tree = DecisionTreeRegressor(data, datasetInfo, responses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree = DecisionTreeRegressor(data, datasetInfo, responses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on mixed categorical data (optionally with instance weights).

---

#### Constructor parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](../load_save.md#loading-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `responses` | [`arma::rowvec`](../matrices.md) | Training responses (e.g. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
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

***Note:*** different types can be used for `data`, `responses`, and `weights`
(e.g., `arma::fmat`, `arma::sp_mat`).  However, the element type of `data`,
`responses`, and `weights` all must match; for example, if `data` has type
`arma::fmat`, then `responses` and `weights` must have type `arma::frowvec`.

### Training

If training is not done as a part of the constructor call, it can be done with
one of the following versions of the `Train()` member function:

 * `tree.Train(data, responses,          minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
 * `tree.Train(data, responses, weights, minLeafSize=10, minGainSplit=1e-7, maxDepth=0)`
   - Train on numerical-only data (optionally with instance weights).

---

 * `tree.Train(data, datasetInfo, responses)`
 * `tree.Train(data, datasetInfo, responses, weights)`
 * `tree.Train(data, datasetInfo, responses,          minLeafSize, minGainSplit, maxDepth)`
 * `tree.Train(data, datasetInfo, responses, weights, minLeafSize, minGainSplit, maxDepth)`
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

### Prediction

Once a `DecisionTreeRegressor` is trained, the `Predict()` member function can
be used to make class predictions for new data.

 * `double predictedValue = tree.Predict(point)`
   - ***(Single-point)***
   - Predict and return the value for a single point.

---

 * `tree.Predict(data, predictions)`
   - ***(Multi-point)***
   - Predict and return values for every point in the given matrix `data`.
   - The predictions for each point are stored in `predictions`, which is set to
     length `data.n_cols`.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

#### Prediction Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for prediction. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for prediction. |
| _multi-point_ | `predictions` | [`arma::rowvec&`](../matrices.md) | Vector to store predictions into. |

***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other Functionality

 * A `DecisionTreeRegressor` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `tree.NumChildren()` will return a `size_t` indicating the number of children
   in the node `tree`.

 * `tree.NumLeaves()` will return the total number of leaf nodes that are
   descendants of the node `tree`.

 * `tree.Child(i)` will return a `DecisionTreeRegressor` object representing the
   `i`th child of the node `tree`.

 * `tree.SplitDimension()` returns a `size_t` indicating which dimension the
   node `tree` splits on.

For complete functionality, the [source
code](/src/mlpack/methods/decision_tree/decision_tree_regressor.hpp) can be
consulted.  Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`DecisionTreeRegressor`.

---

Train a decision tree regressor on mixed categorical data and save the model to
disk.

```c++
// Load a categorical dataset.
arma::mat data;
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/telecom_churn.arff.
mlpack::data::Load("telecom_churn.arff", data, info, true);

arma::rowvec responses;
// See https://datasets.mlpack.org/telecom_churn.responses.csv.
mlpack::data::Load("telecom_churn.responses.csv", responses, true);

// Split data into training set (80%) and test set (20%).
arma::mat trainData, testData;
arma::rowvec trainResponses, testResponses;
mlpack::data::Split(data, responses, trainData, testData, trainResponses,
    testResponses, 0.2);

// Create the tree.
mlpack::DecisionTreeRegressor tree;
// Train on the given dataset, specifying a minimum gain of 1e-6 and keeping the
// default minimum leaf size.
const double mse = tree.Train(trainData, info, trainResponses,
    10 /* minimum leaf size */, 1e-6 /* minimum gain */);
// Print the MSE of the trained tree.
std::cout << "MSE of trained tree is " << mse << "." << std::endl;

// Compute prediction on the first test point.
const double firstPrediction = tree.Predict(testData.col(0));
std::cout << "Predicted value for first test point is " << firstPrediction
    << "." << std::endl;

// Compute predictions on test data.
arma::rowvec testPredictions;
tree.Predict(testData, testPredictions);

// Compute the average error on the test set.
const double testAverageError = arma::mean(testResponses - testPredictions);
std::cout << "Average error on test set: " << testAverageError << "."
    << std::endl;

// Save the tree to "tree.bin" with the name "tree".
mlpack::data::Save("tree.bin", "tree", tree);
```

---

Load a tree and print some information about it.

```c++
mlpack::DecisionTreeRegressor tree;
// This call assumes a tree called "tree" has already been saved to `tree.bin`
// with `data::Save()`.
mlpack::data::Load("tree.bin", "tree", tree, true);

std::cout << "Information about the DecisionTreeRegressor in `tree.bin`:"
    << std::endl;
std::cout << " * The root node has " << tree.NumChildren() << " children."
    << std::endl;
std::cout << " * The tree has " << tree.NumLeaves() << " leaves." << std::endl;
if (tree.NumChildren() > 0)
{
  for (size_t i = 0; i < tree.NumChildren(); ++i)
  {
    std::cout << " * Child " << i << " of the root has "
        << tree.Child(i).NumLeaves() << " leaves in its subtree." << std::endl;
  }
}
```

### Advanced Functionality: Template Parameters

#### Using different element types.

`DecisionTreeRegressor`'s constructors, `Train()`, and `Predict()` functions
support any data type, so long as it supports the Armadillo matrix API.  So, for
instance, learning can be done on single-precision floating-point data:

```c++
// 1000 random points in 10 dimensions.
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random responses for each point, with a normal distribution.
arma::frowvec responses = arma::randn<arma::frowvec>(1000);

// Train in the constructor.
mlpack::DecisionTreeRegressor tree(dataset, responses, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::frowvec predictions;
tree.Predict(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 1) << " test points predicted to have "
    << "value greater than 1." << std::endl;
```

---

#### Fully custom behavior.

The `DecisionTreeRegressor` class also supports several template parameters,
which can be used for custom behavior during learning.  The full signature of
the class is as follows:

```
DecisionTreeRegressor<FitnessFunction,
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
 * The `MSEGain` _(default)_ and `MADGain` classes are available for drop-in
   usage.
 * A custom class must implement three functions:

```c++
// You can use this as a starting point for implementation.
class CustomFitnessFunction
{
  // Compute the gain for the given vector of values, where `values[i]` has an
  // associated instance weight `weights[i]`.
  //
  // `RowType` and `WeightVecType` will be vector types following the Armadillo
  // API.  If `UseWeights` is `false`, then the `weights` vector should be
  // ignored (e.g. the responses are not weighted).
  //
  // In the version with `begin` and `end` parameters, only the subset between
  // `labels[begin]` and `labels[end]` (inclusive) should be considered.
  template<bool UseWeights, typename RowType, typename WeightVecType>
  double Evaluate(const RowType& labels,
                  const WeightVecType& weights);

  template<bool UseWeights, typename RowType, typename WeightVecType>
  double Evaluate(const RowType& labels,
                  const WeightVecType& weights,
                  const size_t begin,
                  const size_t end);

  // Return the output value for prediction for a leaf node whose training
  // values are made up of the values in the vector `responses` (optionally with
  // associated instance weights `weights`).
  //
  // `ResponsesType` and `WeightsType` will be vector types following the
  // Armadillo API.  If `UseWeights` is `false`, then the `weights` vector
  // should be ignored (e.g. the responses are not weighted).
  template<bool UseWeights, typename ResponsesType, typename WeightsType>
  double OutputLeafValue(const ResponsesType& responses,
                         const WeightsType& weights);
};
```

***Note:*** this API differs from the `FitnessFunction` API required for
[`DecisionTree`](decision_tree.md)!

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
  template<bool UseWeights, typename VecType, typename ResponsesType,
      typename WeightVecType>
  double SplitIfBetter(const double bestGain,
                       const VecType& data,
                       const ResponsesType& responses,
                       const WeightVecType& weights,
                       const size_t minLeafSize,
                       const double minGainSplit,
                       arma::vec& splitInfo,
                       AuxiliarySplitInfo& aux,
                       FitnessFunction& function);

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

***Note:*** this API differs from the `NumericSplitType` API required for
[`DecisionTree`](decision_tree.md)!

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
   more than two _classes_.
   - ***Note***: `BestBinaryCategoricalSplit` should not be chosen when there
     are multiple classes and many categories.
   - ***Note***: for regression tasks,
     [W. Fisher's proof of correctness](https://www.mlpack.org/papers/fisher.pdf)
     only applies to when `FitnessFunction` is `MSEGain`; therefore,
     `BestBinaryCategoricalSplit` requires the use of `MSEGain`.
 * A custom class must take a [`FitnessFunction`](#fitnessfunction) as a
   template parameter, implement three functions, and have an internal
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
  template<bool UseWeights, typename VecType, typename ResponsesType,
           typename WeightVecType>
  static double SplitIfBetter(
      const double bestGain,
      const VecType& data,
      const size_t numCategories,
      const ResponsesType& labels,
      const WeightVecType& weights,
      const size_t minLeafSize,
      const double minGainSplit,
      arma::vec& splitInfo,
      AuxiliarySplitInfo& aux,
      FitnessFunction& fitnessFunction);

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

***Note:*** this API differs from the `CategoricalSplitType` API required for
[`DecisionTree`](decision_tree.md)!

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
      constructor of `DecisionTree<>` or the `Train()` function, each random
      subset will be of size `n`.
 * Each `DecisionTreeRegressor` [constructor](#constructors) and each version of
   the [`Train()`](#training) function optionally accept an instantiated
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
