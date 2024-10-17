## `RandomForest`

The `RandomForest` class implements a parallelized random forest classifier that
supports numerical and categorical features, by default using Gini gain to
choose which feature to split on in each tree.

Random forests are a collection of decision trees that give better performance
than a single decision tree.  They are useful for classifying points with
_discrete labels_ (i.e.  `0`, `1`, `2`).  This implementation of the
`RandomForest` class is not for regression (i.e. predicting _continuous
values_).

mlpack's `RandomForest` class offers configurability via template parameters and
runtime parameters.  This is used to provide the additional API-compatible
`ExtraTrees` class.  To use `ExtraTrees`, simply replace `RandomForest` with
`ExtraTrees` in any of the documentation below.  ([More
information...](#fully-custom-behavior))

#### Simple usage example:

```c++
// Train a random forest on random numeric data and predict labels on test data:

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testData(10, 500, arma::fill::randu); // 500 test points.

mlpack::RandomForest rf;            // Step 1: create model.
rf.Train(dataset, labels, 5, 10);   // Step 2: train model.
arma::Row<size_t> predictions;
rf.Classify(testData, predictions); // Step 3: classify points.
// You can also use `ExtraTrees` instead of `RandomForest`!

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `RandomForest` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [`DecisionTree`](decision_tree.md)
 * [`DecisionTreeRegressor`](decision_tree_regressor.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [Random forest on Wikipedia](https://en.wikipedia.org/wiki/Random_forest)
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Leo Breiman's Random Forests page](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)

### Constructors

 * `rf = RandomForest()`
   - Initialize the random forest without training.
   - You will need to call [`Train()`](#training) later to train the tree
     before calling [`Classify()`](#classification).

---

 * `rf = RandomForest(data, labels, numClasses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
 * `rf = RandomForest(data, labels, numClasses, weights, numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
   - Train on numerical-only data (optionally with instance weights).

---

 * `rf = RandomForest(data, info, labels, numClasses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
 * `rf = RandomForest(data, info, labels, numClasses, weights, numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0)`
   - Train on mixed categorical data (optionally with instance weights).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `info` | [`data::DatasetInfo`](../load_save.md#loading-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `weights` | [`arma::rowvec`](../matrices.md) | Instance weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
| `numTrees` | `size_t` | Number of trees to train in the random forest. | `20`
|
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

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `rf.Train(data, labels, numClasses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0, warmStart=false)`
 * `rf.Train(data, labels, numClasses, weights, numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0, warmStart=false)`
   - Train on numerical-only data (optionally with instance weights).
   - Returns a `double` with the average gain of each tree in the random forest.
     By default, this is the Gini gain, unless a different
     [`FitnessFunction` template parameter](#fully-custom-behavior) is
     specified.

---

 * `rf.Train(data, info, labels, numClasses,          numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0, warmStart=false)`
 * `rf.Train(data, info, labels, numClasses, weights, numTrees=20, minLeafSize=1, minGainSplit=1e-7, maxDepth=0, warmStart=false)`
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

### Classification

Once a `RandomForest` is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t predictedClass = rf.Classify(point)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class.

---

 * `rf.Classify(point, prediction, probabilitiesVec)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.
    - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `rf.Classify(data, predictions)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `rf.Classify(data, predictions, probabilities)`
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

 * A `RandomForest` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `rf.NumTrees()` will return a `size_t` indicating the number of trees in the
   random forest.

 * `rf.Tree(i)` will return a [`DecisionTree` object](decision_tree.md)
   representing the `i`th decision tree in the random forest.

For complete functionality, the [source
code](/src/mlpack/methods/random_forest/random_forest.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`RandomForest`.

---

Train a random forest incrementally on random mixed categorical data and save it
to disk:

```c++
// Load a categorical dataset.
arma::mat dataset;
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, true);

// Create the random forest.
mlpack::RandomForest rf;
// Train 10 trees on the given dataset, with a minimum leaf size of 3.
rf.Train(dataset, info, labels, 7 /* classes */, 10 /* trees */,
         3 /* minimum leaf size */);

// Now load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, info, true);

arma::Row<size_t> testLabels;
// See https://datasets.mlpack.org/covertype.test.labels.csv.
mlpack::data::Load("covertype.test.labels.csv", testLabels, true);

// Compute test set accuracy.
arma::Row<size_t> testPredictions;
rf.Classify(testDataset, testPredictions);
double accuracy = 100.0 * ((double) arma::accu(testPredictions == testLabels)) /
    testLabels.n_elem;
std::cout << "After training 10 trees, test set accuracy is " << accuracy
    << "%." << std::endl;

// Now train another 10 trees and compute the test accuracy.
rf.Train(dataset, info, labels, 7 /* classes */, 10 /* trees */,
         3 /* minimum leaf size */, 0.0 /* minimum split gain */,
         0 /* maximum depth (unlimited) */, true /* incremental training */);

rf.Classify(testDataset, testPredictions);
accuracy = 100.0 * ((double) arma::accu(testPredictions == testLabels)) /
    testLabels.n_elem;
std::cout << "After training 20 trees, test set accuracy is " << accuracy
    << "%." << std::endl;

// Save the random forest to disk.
mlpack::data::Save("rf.bin", "rf", rf);
```

---

Load a random forest and print some information about it.

```c++
mlpack::RandomForest rf;
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

Train a random forest on categorical data, and compare its performance with the
performance of each individual tree:

```c++
// Load a categorical dataset (training and test sets).
arma::mat dataset, testDataset;
mlpack::data::DatasetInfo info;
arma::Row<size_t> labels, testLabels;

// See the following files:
//  * https://datasets.mlpack.org/covertype.train.arff
//  * https://datasets.mlpack.org/covertype.train.labels.csv
//  * https://datasets.mlpack.org/covertype.test.arff
//  * https://datasets.mlpack.org/covertype.test.labels.csv
mlpack::data::Load("covertype.train.arff", dataset, info, true);
mlpack::data::Load("covertype.train.labels.csv", labels, true);
mlpack::data::Load("covertype.test.arff", testDataset, info, true);
mlpack::data::Load("covertype.test.labels.csv", testLabels, true);

// Create the random forest.
mlpack::RandomForest rf;
// Train 20 trees on the given dataset, with a minimum leaf size of 5.
rf.Train(dataset, info, labels, 7 /* classes */, 20 /* trees */,
         5 /* minimum leaf size */);

// Compute test set accuracy for each tree.
arma::Row<size_t> testPredictions;
for (size_t i = 0; i < rf.NumTrees(); ++i)
{
  rf.Tree(i).Classify(testDataset, testPredictions);
  const double accuracy = 100.0 *
      ((double) arma::accu(testPredictions == testLabels)) / testLabels.n_elem;
  std::cout << "Tree " << i << " has test accuracy " << accuracy << "%."
      << std::endl;
}

// Now compute accuracy using the whole forest.
rf.Classify(testDataset, testPredictions);
const double accuracy = 100.0 *
    ((double) arma::accu(testPredictions == testLabels)) / testLabels.n_elem;
std::cout << "The whole forest has test accuracy " << accuracy << "%."
    << std::endl;
```

---

Train an `ExtraTrees` model on random numeric data.

```c++
// 1000 random points in 10 dimensions.
arma::mat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor, using 10 trees in the forest.
// Note that `ExtraTrees` has exactly the same API as `RandomForest`.
mlpack::ExtraTrees<> rf(dataset, labels, 5, 10);

// Create a single test point.
arma::vec testPoint(10, arma::fill::randu);

size_t prediction;
arma::vec probabilities;
rf.Classify(testPoint, prediction, probabilities);
std::cout << "Test point predicted to be class " << prediction << "."
    << std::endl;
std::cout << "Probabilities of each class: " << probabilities.t();
```

---

See also the following fully-working examples:

 - [Rainfall prediction with `RandomForest`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/random_forest/rainfall_prediction/rainfall-prediction-cpp.ipynb)
 - [Forest cover type prediction with `RandomForest`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/random_forest/forest_covertype_prediction/covertype-rf-cpp.ipynb)

### Advanced Functionality: Template Parameters

#### Using different element types.

`RandomForest`'s constructors, `Train()`, and `Predict()` functions support any
data type, so long as it supports the Armadillo matrix API.  So, for instance,
learning can be done on single-precision floating-point data:

```c++
// 1000 random points in 10 dimensions.
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
mlpack::RandomForest rf(dataset, labels, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
rf.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 0) << " test points classified as class "
    << "0." << std::endl;
```

---

#### Fully custom behavior.

mlpack provides a few variants of the random forest classifier, using the
template parameters of the `RandomForest` class.  The following types can be
used as drop-in replacements throughout this documentation page:

 * `RandomForest`
    - This is an implementation of Breiman's seminal random forest algorithm
      ([website](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm),
      [paper pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)).
    - The [`DecisionTree`](decision_tree.md) class is used for each individual
      decision tree.
    - When training each individual decision tree, bootstrapping is used to
      compute the samples given to each tree for training.

 * `ExtraTrees`
    - This is an implementation of the Extremely Randomized Trees algorithm
      ([paper pdf](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=336a165c17c9c56160d332b9f4a2b403fccbdbfb)).
    - When training an `ExtraTrees` model, each individual decision tree chooses
      splits for numeric data randomly.
    - Training an `ExtraTrees` model is generally much faster than
      `RandomForest`, but the accuracy of the `ExtraTrees` model will be lower.
    - To use `ExtraTrees`, simply replace `RandomForest` with `ExtraTrees` in
      the documentation below.

---

Fully custom classes can also be used to control the behavior of the
`RandomForest` class.  The full signature of the class is as follows:

```
RandomForest<FitnessFunction,
             DimensionSelectionType,
             NumericSplitType,
             CategoricalSplitType,
             UseBootstrap>
```

 * `FitnessFunction`: the measure of goodness to use when deciding on tree
   splits
 * `DimensionSelectionType`: the strategy used for proposing dimensions to
   attempt to split on
 * `NumericSplitType`: the strategy used for finding splits on numeric data
   dimensions
 * `CategoricalSplitType`: the strategy used for finding splits on categorical
   data dimensions
 * `UseBootstrap`: a boolean indicating whether or not to use a bootstrap sample
   when training each tree in the forest

Note that the first four of these template parameters are exactly the same as
the template parameters for the
[`DecisionTree`](decision_tree.md#fully-custom-behavior) class.

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

#### `DimensionSelectionType`

 * When splitting a tree in the forest, `DimensionSelectionType` proposes
   possible dimensions to try splitting on.
 * `MultipleRandomDimensionSelect` _(default)_ is available for drop-in usage
   and proposes a different random subset of dimensions at each decision tree
   node.
    - By default each random subset is of size `sqrt(d)` where `d` is the number
      of dimensions in the data.
    - If constructed as `MultipleRandomDimensionSelect(n)` and passed to the
      constructor of `RandomForest` or the `Train()` function, each random
      subset will be of size `n`.
 * Each `RandomForest` [constructor](#constructors) and each version of
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
   the gain.  (Used by the `ExtraTrees` [variant](#fully-custom-behavior).)
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
 * A custom class must take a [`FitnessFunction`](#fitnessfunction) as a
   template parameter, implement three functions, and have an internal structure
   `AuxiliarySplitInfo` that is used at classification time:

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
      const size_t minLeafSize,
      const double minGainSplit,
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

#### `UseBootstrap`

 * A `bool` value that indicates whether or not a bootstrap sample of the
   dataset should be used for the training of each individual decision tree in
   the random forest.
 * If `true` _(default)_, a different bootstrap sample of the same size as the
   dataset will be used to train each decision tree.
 * If `false` _(default for the `ExtraTrees` [variant](#fully-custom-behavior))_, the full
   dataset will be used to train each decision tree.
