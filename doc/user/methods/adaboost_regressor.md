## `AdaBoostRegressor`

The `AdaBoostRegressor` class implements the 'adaptive boosting' regressor 
AdaBoost.R2.
An AdaBoost regressor is a meta-estimator that begins by fitting a regressor 
on the original dataset and then fits additional copies of the regressor on the
same dataset but where the weights of instances are adjusted according to the 
error of the current prediction. As such, subsequent regressors focus more on 
difficult cases.

`AdaBoostRegressor` is useful for regressions; i.e., predicting 
_continuous values_ (`0.2`, `1.3`, etc.). For predicting _discrete label_
(classification), see [`AdaBoost`](adaboost.md)

mlpack's `AdaBoostRegressor` class offers configurability via template
parameters.

#### Simple usage example:

```c++
//Train a adaboost regressor on random numeric data and make prediction.

//All data and responses are uniform random; this uses 10 dimensional data.
//Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::rowvec responses = arma::randn<arma::rowvec>(1000);
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::AdaBoostRegressor<> abr;     // Step 1: create model.
abr.Train(dataset, responses);         // Step 2: train model
arma::rowvec predictions;
abr.Predict(testDataset, predictions); // Step 3: use model to predict.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 0.7) << " test points predicted to have"
    << " responses greater than 0.7." << std::endl;
std::cout << arma::accu(predictions < 0) << " test points predicted to have "
    << "negative responses." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `AdaBoostRegressor` objects.
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
 * [AdaBoost](adaboost.md)
 * [AdaBoost on Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Improving Regressors using Boosting Techniques](https://dl.acm.org/doi/10.5555/645526.657132)

### Constructors
 * `ab = AdaBoostRegressor()`
   - Initialize the AdaBoost regressor without training.
   - You will need to call [`Train()`](#training) later to train the tree 
     before calling [`Predict()`](#prediction).
  
---

  * `ab =  AdaBoostRegressor(dataset, responses, numTrees = 20, minimumLeafSize = 10, minimumGainSplit = 1e-7, maximumDepth = 4)`
    - Train on numerical-only data.

---

  * `ab =  AdaBoostRegressor(dataset, datasetInfo, responses, numTrees = 20, minimumLeafSize = 10, minimumGainSplit = 1e-7, maximumDepth = 4)`
    - Train on mixed categorical data.

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](../../tutorials/datasetmapper.md) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `responses` | [`arma::rowvec`]('../matrices.md') | Training responses (eg. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numTrees` | `size_t` | Number of trees to train in the adaboost regressor. | `20` |
| `minimumLeafSize` | `size_t` | Minimum number of points in each leaf node of each decision tree. | `10` |
| `minimumGainSplit` | `double` | Minimum gain for a node to split in each decision tree. | `1e-7` |
| `maximumDepth` | `size_t` | Maximum depth for each decision tree. (0 means no limit.) | `4` |

### Training

If traning is not done as part of the constructor call, it can be done with one 
of the following versions of the `Train()` member function:

  * `ab.Train(data, responses, numTrees=20, minimumLeafSize=10, minimumGainSplit=1e-7, maximumDepth=4)`
     - Train on numerical-only data.

---

  * `ab.Train(data, datasetInfo, responses, numTrees=20, minimumLeafSize=10, minimumGainSplit=1e-7, maximumDepth=4)`
     - Train on mixed categorical data.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

### Prediction

Once a `AdaBoostRegressor` is trained, the `Predict()` member function can 
be used to make class predictions for new data.

 * `double predictedValue = ab.Predict(point)`
   - ***(Single-point)***
   - Predict and return the value for a single point.

---

 * `ab.Predict(data, predictions)`
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

 * A `AdaBoostRegressor` can be serialized with [`data::Save()`](../formats.md) and
   [`data::Load()`](../formats.md).

 * `ab.NumTrees()` will return a `size_t` indicating the number of trees in the
   adaboost regressor.

 * `ab.Tree(i)` will return a [`DecisionTreeRegressor` object](decision_tree_regressor.md)
   representing the `i`th decision tree in the adaboost regressor.

For complete functionality, the [source
code](/src/mlpack/methods/adaboost/adaboost_regressor.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of 
`AdaBoostRegressor`.

---

Train a adaboost regressor on mixed categorical data.

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
mlpack::AdaBoostRegressor<> adr;
// Train on the given dataset, specifying a minimum gain of 1e-7 and keeping the
// default minimum leaf size.
const double mse = adr.Train(trainData, info, trainResponses, 20 /*number of trees*/
    ,10 /* minimum leaf size */, 1e-7 /* minimum gain */, 4 /*maximum depth*/);
// Print the MSE of the trained tree.
std::cout << "MSE of trained tree is " << mse << "." << std::endl;

// Compute prediction on the first test point.
const double firstPrediction = adr.Predict(testData.col(0));
std::cout << "Predicted value for first test point is " << firstPrediction
    << "." << std::endl;

// Compute predictions on test data.
arma::rowvec testPredictions;
adr.Predict(testData, testPredictions);

// Compute the average error on the test set.
const double testAverageError = arma::mean(testResponses - testPredictions);
std::cout << "Average error on test set: " << testAverageError << "."
    << std::endl;
```

---

Load a adaboost regressor and print some information about it.

```c++
mlpack::AdaBoostRegressor<> adr;
// This call assumes a adaboost regressor called "adr" has already been saved to
// `adr.bin` with `data::Save()`.
mlpack::data::Load("adr.bin", "adr", adr, true);

std::cout << "The adaboost regressor in 'adr.bin' contains " << adr.NumTrees()
    << " trees." << std::endl;
if (adr.NumTrees() > 0)
{
  std::cout << "The first tree's root node has " << adr.Tree(0).NumChildren()
      << " children." << std::endl;
}
```

---

Train a adaboost regressor on categorical data, and compare its performance 
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
mlpack::AdaBoostRegressor<> adr;
// Train on the given dataset, specifying a minimum gain of 1e-7 and keeping the
// default minimum leaf size.
const double mse = adr.Train(trainData, info, trainResponses, 20 /*number of trees*/
    ,10 /* minimum leaf size */, 1e-7 /* minimum gain */, 4 /*maximum depth*/);

// Compute test accuracy for each tree
arma::rowvec testPredictions;
for(size_t i = 0; i<adr.NumTrees(); i++)
{
  adr.Tree(i).Predict(testData, testPredictions);
  const double testAverageError = arma::mean(testResponses - testPredictions);
  std::cout << "Tree " << i << " has averageError: " << testAverageError << "."
    << std::endl;
}

// Compute predictions on test data using whole forest.
adr.Predict(testData, testPredictions);

// Compute the average error on the test set.
const double testAverageError = arma::mean(testResponses - testPredictions);
std::cout << "The whole forest has Average error: " << testAverageError << "."
    << std::endl;
```

---
### Advanced Functionality: Template Parameters

#### Fully custom behavior.

The `AdaBoostRegressor` class also supports several template parameters, 
which can be used for custom behavior during learning. The full signature of 
the class is as follow:

```c++
AdaBoostRegressor<LossFunctionType,
                  FitnessFunction,
                  DimensionSelectionType,
                  NumericSplitType,
                  CategoricalSplitType>
```
 * `LossFunctionType` : the type of loss function needs to be used.
 * `FitnessFunction` : the measure of goodness to use when deciding on tree
   splits
 * `DimensionSelectionType`: the strategy used for proposing dimensions to
   attempt to split on
 * `NumericSplitType`: the strategy used for finding splits on numeric data
   dimensions
 * `CategoricalSplitType`: the strategy used for finding splits on categorical
   data dimensions

Below, details are given for the requirements of each of these template types.

---
#### `LossFunctionType`

  * Specifies the loss function to use when learning a decision tree.
  * The `LinearLoss` _(default)_, `SquareLoss` and `ExponentialLoss` are
    available for drop-in usage.
  * A custom class must implement a function:

```c++
// You can use this as a starting point for implementation.
class CustomLossFunction
{
  public:
  // Compute the loss vector of the error_vec vector
  // and return the loss vector
  template <typename VecType>
  static VecType Calculate (const VecType& error_vec);
};
```

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
      constructor of `AdaBoostRegressor` or the `Train()` function, each 
      random subset will be of size `n`.
 * Each `AdaBoostRegressor` [constructor](#constructors) and each version of
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