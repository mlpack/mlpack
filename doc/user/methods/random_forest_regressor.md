## `RandomForestRegressor`

The `RandomForestRegressor` class implements a parallelized random forest regressor 
that supports numerical and categorical features, by default using MSE (minimum 
sqaured error) to chosee which feature to split on.

Random forests are a collection of decision trees that give better performance than
a single decision tree. They are useful for regressions; i.e., predicting 
_continuous values_ (`0.3`, `1.2`, etc.). For predicting _discrete labels_ 
(classification), see [`RandomForest`](random_forest.md)

mlpacks's `RandomForestRangressor` class offers configurability via template 
parameters.

#### Simple usage example:

```c++
//Train a random forest on random numeric data and make prediction.

//All data and responses are uniform random; this uses 10 dimensional data.
//Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::rowvec responses = arma::randn<arma::rowvec>(1000);
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::RandomForestRegressor rf;     // Step 1: create model.
rf.Train(dataset, responses);         // Step 2: train model
arma::rowvec predictions;
rf.Predict(testDataset, predictions); // Step 3: use model to predict.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 0.7) << " test points predicted to have"
    << " responses greater than 0.7." << std::endl;
std::cout << arma::accu(predictions < 0) << " test points predicted to have "
    << "negative responses." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `RandomForestRegressor` objects.
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
 * [`DecisionTreeRegressor`](decision_tree_regressor.md)
 * [Random forest](random_forest.md)
 * [mlpack classifiers](#mlpack_regression_techniques) <!-- TODO: fix link! -->
 * [Random forest on Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Leo Breiman's Random Forests page](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)

### Constructors

 * `rf = RandomForestRegressor()`
   - Initialize the random forest regressor without training.
   - You will need to call [`Train()`](#training) later to train the tree 
     before calling [`Predict()`](#prediction).

---

 * `rf = RandomForestRegressor(data, responses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
 * `rf = RandomForestRegressor(data, responses, weights, numtrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
     - Train on numerical-only data (optionally with instance weights).

---

 * `rf = RandomForestRegressor(data, datasetInfo, responses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
 * `rf = RandomForestRegressor(data, datasetInfo, responses, weights, numtrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
     - Train on mixed categorical data (optionally with instance weights).


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
| `info` | [`data::DatasetInfo`](../../tutorials/datasetmapper.md) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `responses` | [`arma::rowvec`]('../matrices.md') | Training responses (eg. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`]('../matrices.md') | Instance weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
| `numTrees` | `size_t` | Number of trees to train in the random forest. | `20` |
| `minLeafSize` | `size_t` | Minimum number of points in each leaf node of each decision tree. | `1` |
| `minGainSplit` | `double` | Minimum gain for a node to split in each decision tree. | `1e-7` |
| `maxDepth` | `size_t` | Maximum depth for each decision tree. (0 means no limit.) | `0` |
| `warmStart` | `bool` | (Only available in `Train()`.)  If true, training adds `numTrees` trees to the random forest.  If `false`, an entirely new random forest will be created. | `false` |

 * If OpenMP is enabled<!-- TODO: link! -->, one thread will be used to train
   each of the `numTrees` trees in the random forest.  The computational effort
   involved with training a random forest increases linearly with the number of
   trees.
 * The default `minLeafSize` is `1`, unlike `DecisionTree`.  This is because
   random forests are less susceptible to overfitting due to their ensembled
   nature.
 * Note that the default `minLeafSize` of `1` will make large decision trees,
   and so if a smaller-sized model is desired, this value should be increased
   (at the potential cost of accuracy).
 * `minGainSplit` can also be increased if a smaller-sized model is desired.

***Note:*** different types can be used for `data` and `weights` (e.g.,
`arma::fmat`, `arma::sp_mat`).  However, the element type of `data` and
`weights` must match; for example, if `data` has type `arma::fmat`, then
`weights` must have type `arma::frowvec`.

### Training

If traning is not done as part of the constructor call, it can be done with one 
of the following versions of the `Train()` member function:

 * `rf.Train(data, responses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
 * `rf.Train(data, responses, weights, numtrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
     - Train on numerical-only data (optionally with instance weights).

---

 * `rf.Train(data, datasetInfo, responses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
 * `rf.Train(data, datasetInfo, responses, weights, numtrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
     - Train on mixed categorical data (optionally with instance weights).

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

**Notes**:

 * The `warmStart` option, which allows incremental training (i.e. additional
   training on top of an existing model) is of type `bool` and defaults to
   `false`.  This option is not available in the [constructors](#constructors).

 * `Train()` returns a `double` with the average gain of each tree in the random
   forest.  By default, this is the Gini gain, unless a different
   [`FitnessFunction` template parameter](#fully-custom-behavior) is specified.


### Prediction

Once a `RandomForestRegressor` is trained, the `Predict()` member function can 
be used to make class predictions for new data.

 * `double predictedValue = rf.Predict(point)`
   - ***(Single-point)***
   - Predict and return the value for a single point.

---

 * `rf.Predict(data, predictions)`
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
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md) points for prediction. |
| _multi-point_ | `predictions` | [`arma::rowvec&`](../matrices.md) | Vector to store predictions into. |

***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other Functionality

<!-- TODO: we should point directly to the documentation of those functions -->

 * A `RandomForestRegressor` can be serialized with [`data::Save()`](../formats.md) and
   [`data::Load()`](../formats.md).

 * `rf.NumTrees()` will return a `size_t` indicating the number of trees in the
   random forest.

 * `rf.Tree(i)` will return a [`DecisionTreeRegressor` object](decision_tree_regressor.md)
   representing the `i`th decision tree in the random forest.

For complete functionality, the [source
code](/src/mlpack/methods/random_forest/random_forest_regressor.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of 
`RandomForestRegressor`.

---

Train a random forest regressor on mixed categorical data.

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
mlpack::RandomForestRegressor rf;
// Train on the given dataset, specifying a minimum gain of 1e-10 and keeping the
// default minimum leaf size.
const double mse = rf.Train(trainData, info, trainResponses, 20 /*number of trees*/
    ,1 /* minimum leaf size */, 1e-10 /* minimum gain */);
// Print the MSE of the trained tree.
std::cout << "MSE of trained tree is " << mse << "." << std::endl;

// Compute prediction on the first test point.
const double firstPrediction = rf.Predict(testData.col(0));
std::cout << "Predicted value for first test point is " << firstPrediction
    << "." << std::endl;

// Compute predictions on test data.
arma::rowvec testPredictions;
rf.Predict(testData, testPredictions);

// Compute the average error on the test set.
const double testAverageError = arma::mean(testResponses - testPredictions);
std::cout << "Average error on test set: " << testAverageError << "."
    << std::endl;
```

---

Load a random forest regressor and print some information about it.

```c++
mlpack::RandomForestRegressor rf;
// This call assumes a random forest called "rf" has already been saved to
// `rf.bin` with `data::Save()`.
mlpack::data::Load("rf.bin", "rf", rf, true);

std::cout << "The random forest in 'rf.bin' contains " << rf.NumTrees()
    << " trees." << std::endl;
if (rf.NumTrees() > 0)
{
  std::cout << "The first tree's root node has " << rf.Tree(0).NumChildren()
      << " children." << std::endl;
}
```

---

Train a random forest regressor on categorical data, and compare its performance 
with the performance of each individual tree:

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
mlpack::RandomForestRegressor rf;
// Train on the given dataset, specifying a minimum gain of 1e-10 and keeping the
// default minimum leaf size.
const double mse = rf.Train(trainData, info, trainResponses, 20 /*number of trees*/
    ,1 /* minimum leaf size */, 1e-10 /* minimum gain */);

// Compute test accuracy for each tree
arma::rowvec testPredictions;
for(size_t i = 0; i<rf.NumTrees(); i++)
{
  rf.Tree(i).Predict(testData, testPredictions);
  const double testAverageError = arma::mean(testResponses - testPredictions);
  std::cout << "Tree " << i << " has averageError: " << testAverageError << "."
    << std::endl;
}

// Compute predictions on test data using whole forest.
rf.Predict(testData, testPredictions);

// Compute the average error on the test set.
const double testAverageError = arma::mean(testResponses - testPredictions);
std::cout << "The whole forest has Average error: " << testAverageError << "."
    << std::endl;
```

### Advanced Functionality: Template Parameters

#### Using different element types.

`RandomForestRegressor`'s contructors, `Train()`, and `Predict()` functions support 
any data type, so long as it supports the Armadillo matrix API. So, for instance, learning can be done on single-precision floating-point data:

```c++
// 1000 random points in 10 dimensions.
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random responses for each point, with a normal distribution.
arma::frowvec responses = arma::randn<arma::frowvec>(1000);

// Train in the constructor.
mlpack::RandomForestRegressor rf(dataset, responses);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::frowvec predictions;
rf.Predict(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 1) << " test points predicted to have "
    << "value greater than 1." << std::endl;
```

---

#### Fully custom behavior.

The `RandomForestRegressor` class also supports several template parameters, 
which can be used for custom behavior during learning. The full signature of 
the class is as follow:

```c++
RandomForestRegressor<FitnessFunction,
                      DimensionSelectionType,
                      NumericSplitType,
                      CategoricalSplitType,
                      UseBootstrap>
```

 * `FitnessFunction` : the measure of goodness to use when deciding on tree
   splits
 * `DimensionSelectionType`: the strategy used for proposing dimensions to
   attempt to split on
 * `NumericSplitType`: the strategy used for finding splits on numeric data
   dimensions
 * `CategoricalSplitType`: the strategy used for finding splits on categorical
   data dimensions
 * `UseBootstrap`: a boolean indicating whether or not to use a bootstrap sample
   when training each tree in the forest

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
[`DecisionTree`](decision_tree.md)! <!-- TODO: fix link! -->

---

#### `DimensionSelectionType`

 * When splitting a decision tree, `DimensionSelectionType` proposes possible
   dimensions to try splitting on.
 * `MultipleRandomDimensionSelect` _(default)_ is available for drop-in usage
   and proposes a different random subset of dimensions at each decision tree
   node.
    - By default each random subset is of size `sqrt(d)` where `d` is the number
      of dimensions in the data.
    - If constructed as `MultipleRandomDimensionSelect(n)` and passed to the
      constructor of `RandomForestRegressor` or the `Train()` function, each 
      random subset will be of size `n`.
 * Each `RandomForestRegressor` [constructor](#constructors) and each version of
   the [`Train()`](#training) function optionally accept an instantiated
   `DimensionSelectionType` object as the very last parameter (after
   `maxDepth` in the constructor, or `warmStart` in `Train()`), in case some
   internal state in the dimension selection mechanism is required.
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
   [`RandomForest`](random_forest.md).) <!-- TODO: fix link! -->
 * A custom class must take a [`FitnessFunction`](#fitness-function) as a
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

***Note:*** this API differs from the `NumericSplitType` API required for
[`DecisionTree`](decision_tree.md)! <!-- TODO: fix link! -->

---

#### `CategoricalSplitType`

 * Specifies the strategy to be used during training when splitting a
    categorical feature.
 * The `AllCategoricalSplit` _(default)_ is available for drop-in usage and
   splits all categories into their own node.
 * A custom class must take a [`FitnessFunction`](#fitness-function) as a
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
  // `splitInfo` should be set to a vector of length 1.  The format of `aux` is
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

***Note:*** this API differs from the `CategoricalSplitType` API required for
[`DecisionTree`](decision_tree.md)! <!-- TODO: fix link! -->

---

#### `UseBootstrap`

 * A `bool` value that indicates whether or not a bootstrap sample of the
   dataset should be used for the training of each individual decision tree in
   the random forest.
 * If `true` _(default)_, a different bootstrap sample of the same size as the
   dataset will be used to train each decision tree.