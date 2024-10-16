## `HoeffdingTree`

The `HoeffdingTree` class implements a streaming (or incremental) decision tree
classifier that supports numerical and categorical features, by default using
Gini impurity to choose which feature to split on.  The class offers several
template parameters and several runtime options that can be used to control the
behavior of the tree.

Hoeffding trees (also known as "Very Fast Decision Trees" or VFDTs) are useful
for classifying points with _discrete labels_ (i.e.  `0`, `1`, `2`).

#### Simple usage example:

```c++
// Train a Hoeffding tree on random numeric data; predict labels on test data:

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::HoeffdingTree tree;              // Step 1: create model.
tree.Train(dataset, labels, 5);          // Step 2a: train model (batch).
tree.Train(dataset.col(0), labels[0]);   // Step 2b: train model (incremental).
arma::Row<size_t> predictions;
tree.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `HoeffdingTree` objects.
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
 * [Random forests](random_forest.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [Incremental decision tree on Wikipedia](https://en.wikipedia.org/wiki/Incremental_decision_tree)
 * [Mining High-Speed Data Streams (pdf)](https://dl.acm.org/doi/pdf/10.1145/347090.347107)

### Constructors

 * `tree = HoeffdingTree()`
   - Initialize tree without training.
   - You will need to call the batch version of [`Train()`](#training) later to
     train the tree before calling [`Classify()`](#classification).

---

 * `tree = HoeffdingTree(dimensionality, numClasses)`
 * `tree = HoeffdingTree(dimensionality, numClasses, successProbability=0.95, maxSamples=0, checkInterval=100, minSamples=100)`
   - Initialize tree for incremental training on numerical-only data.
   - The single-point [`Train()`](#training) function can be used to train
     incrementally.

---

 * `tree = HoeffdingTree(datasetInfo, numClasses)`
 * `tree = HoeffdingTree(datasetInfo, numClasses, successProbability=0.95, maxSamples=0, checkInterval=100, minSamples=100)`
   - Initialize tree for incremental training on mixed categorical data.
   - The single-point [`Train()`](#training) function can be used to train
     incrementally.

---

 * `tree = HoeffdingTree(data, labels, numClasses)`
 * `tree = HoeffdingTree(data, labels, numClasses, batchTraining=true, successProbability=0.95, maxSamples=0, checkInterval=100, minSamples=100)`
   - Train non-incrementally on the given data.
   - The tree will be reset if `numClasses` or the data's dimensionality does
     not match the current settings of the tree.

---

 * `tree = HoeffdingTree(data, datasetInfo, labels, numClasses)
 * `tree = HoeffdingTree(data, datasetInfo, labels, numClasses, batchTraining=true, successProbability=0.95, maxSamples=0, checkInterval=100, minSamples=100)`
   - Train non-incrementally on mixed categorical data.
   - The tree will be reset if `numClasses` or `datasetInfo` does not match the
     current settings of the tree.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](../load_save.md#loading-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `dimensionality` | `size_t` | When using on numeric-only data, this specifies the number of dimensions in the data. | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `batchTraining` | `bool` | If `true`, a batch training algorithm is used, instead of the usual incremental algorithm.  This is generally more efficient for larger datasets. | `true` |
| `successProbability` | `double` | Probability of success required for Hoeffding bound before a node split can happen. | `0.95` |
| `maxSamples` | `size_t` | Maximum number of samples before a node split is forced.  `0` means no limit. | `0` |
| `checkInterval` | `size_t` | Number of samples required before each split check.  Higher values check less often, which is more efficient, but may not split a node as early as possible. | `100` |
| `minSamples` | `size_t` | Minimum number of samples for a node to see before a split is allowed. | `100` |

As an alternative to passing hyperparameters, these can be set with a
standalone method.  The following functions can be used before calling
`Train()`:

 * `tree.SuccessProbability(successProbability);` will set the required success
   probability for splitting to `successProbability`.
 * `tree.MaxSamples(maxSamples);` will set the maximum number of samples before
   a split to `maxSamples`.
 * `tree.CheckInterval(checkInterval);` will set the number of samples between
   split checks to `checkInterval`.
 * `tree.MinSamples(minSamples);` will set the minimum number of samples before
   a split to `minSamples`.

***Notes:***

 * Setting `successProbability` higher than the default means that the Hoeffding
   tree is less likely (and will take more samples) to split a node.  This can
   result in a smaller tree.

 * Different types can be used for `data` (e.g., `arma::fmat`, `arma::sp_mat`).
   See [template parameters](#advanced-functionality-template-parameters) for
   using different `NumericSplitType`s that accept different element types.

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `tree.Train(point, label)`
   - Streaming (incremental) training: train on a single data point.
   - The number of classes and dataset information must have already been
     specified by a previous constructor, `Train()`, or `Reset()` call.

---

 * `tree.Train(data, labels)`
 * `tree.Train(data, labels, numClasses)`
 * `tree.Train(data, labels, numClasses, batchTraining=true, successProbability=0.95, maxSamples=0, checkInterval=100, minSamples=100)`
   - Train on the given data.
   - If the data is mixed categorical, then `datasetInfo` should have already
     been passed via a previous constructor, `Train()`, or `Reset()` call.
   - `numClasses` does not need to be specified if it has been specified in an
     earlier constructor or `Train()` call.

---

 * `tree.Train(data, datasetInfo, labels)`
 * `tree.Train(data, datasetInfo, labels, numClasses)`
 * `tree.Train(data, datasetInfo, labels, numClasses, batchTraining=true, successProbability=0.95, maxSamples=0, checkInterval=100, minSamples=100)`
   - Train on mixed categorical data.
   - The previous overload (without `datasetInfo`) can be used instead if
     `datasetInfo` has already been passed in a previous constructor, `Train()`,
     or `Reset()` call, and has not changed.
   - `numClasses` does not need to be specified if it has been specified in an
     earlier constructor or `Train()` call.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes***:

 * Training is incremental.  Successive calls to `Train()` will train the
   Hoeffding tree further.  To reset the tree, call
   [`Reset()`](#other-functionality).

### Classification

Once a `DecisionTree` is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t predictedClass = tree.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `tree.Classify(point, prediction, probability)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.
    - The probability of class `i` is stored in `probability`.

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
    - The probability of class `predictions[i]` for data point `i` can be
      accessed with `probabilities[i]`.

---

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
| _single-point_ | `probability` | `double&` | `double` to store predicted class probability into. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into.  Will be set to length `data.n_cols`. |
| _multi-point_ | `probabilities` | [`arma::rowvec&`](../matrices.md) | Vector to store probability of predicted class in for each point.  Will be set to length `data.n_cols`. |

***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other Functionality

 * A `HoeffdingTree` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `tree.NumChildren()` will return a `size_t` indicating the number of children
   in the node `tree`.

 * `tree.NumDescendants()` will return a `size_t` indicating the total number of
   descendant nodes of the tree.

 * `tree.Child(i)` will return a `HoeffdingTree` object representing the `i`th
   child of the node `tree`.

 * `tree.SplitDimension()` returns a `size_t` indicating which dimension the
   node `tree` splits on.

 * `tree.NumSamples()` returns a `size_t` indicating the number of points seen
   so far by `tree`, if `tree` has not yet split.  If `tree` has split (i.e. if
   `tree.NumChildren() > 0`), then what is returned is the number of points seen
   up until the split occurred.

 * `tree.NumClasses()` returns a `size_t` indicating the number of classes the
   tree was trained on.

 * `tree.Reset()` will reset the tree to an empty tree, and:
   - `tree.Reset()` will leave the number of classes and dataset information
     (e.g. `datasetInfo`) intact.
   - `tree.Reset(dimensionality, numClasses)` will set the number of classes to
     `numClasses` and set the dimensionality of the data to `dimensionality`,
     assuming all dimensions are numeric.
   - `tree.Reset(datasetInfo, numClasses)` will set the number of classes to
     `numClasses` and set the dataset information to `datasetInfo`.

For complete functionality, the [source
code](/src/mlpack/methods/hoeffding_trees/hoeffding_tree.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`HoeffdingTree`.

---

Train a Hoeffding tree incrementally on mixed categorical data:

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
mlpack::HoeffdingTree tree(info, 7 /* classes */);

// Train on each point in the given dataset.
for (size_t i = 0; i < dataset.n_cols; ++i)
  tree.Train(dataset.col(i), labels[i]);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, info, true);

// Predict class of first test point.
const size_t firstPrediction = tree.Classify(testDataset.col(0));
std::cout << "First test point has predicted class " << firstPrediction << "."
    << std::endl;

// Predict class and probabilities of second test point.
size_t secondPrediction;
double secondProbability;
tree.Classify(testDataset.col(1), secondPrediction, secondProbability);
std::cout << "Second test point has predicted class " << secondPrediction
    << " with probability " << secondProbability << "." << std::endl;
```

---

Train a Hoeffding tree on blocks of a dataset, print accuracy measures on a test
set during training, and save the model to disk.

```c++
// Load a categorical dataset.
arma::mat dataset;
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, true);

// Also load test data.

// See https://datasets.mlpack.org/covertype.test.arff.
arma::mat testDataset;
mlpack::data::Load("covertype.test.arff", testDataset, info, true);

// See https://datasets.mlpack.org/covertype.test.labels.csv.
arma::Row<size_t> testLabels;
mlpack::data::Load("covertype.test.labels.csv", testLabels, true);

// Create the tree with custom parameters.
mlpack::HoeffdingTree tree(info, 7 /* number of classes */);
tree.SuccessProbability(0.99);
tree.CheckInterval(500);

// Now iterate over 10k-point chunks in the dataset.
for (size_t start = 0; start < dataset.n_cols; start += 10000)
{
  size_t end = std::min(start + 9999, (size_t) dataset.n_cols - 1);

  tree.Train(dataset.cols(start, end), info, labels.subvec(start, end));

  // Compute accuracy on the test set.
  arma::Row<size_t> predictions;
  tree.Classify(testDataset, predictions);
  const double accuracy = 100.0 * arma::accu(predictions == testLabels) /
      testLabels.n_elem;

  std::cout << "Accuracy after " << (end + 1) << " points: " << accuracy
      << "\%." << std::endl;
}

// Save the fully trained tree in `tree.bin` with name `tree`.
mlpack::data::Save("tree.bin", "tree", tree, true);
```

---

Load a tree and print some information about it.

```c++
mlpack::HoeffdingTree tree;
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

Train a tree, reset a tree, and train again.

```c++
// See the following files:
//  - https://datasets.mlpack.org/covertype.train.arff.
//  - https://datasets.mlpack.org/covertype.train.labels.csv.
//  - https://datasets.mlpack.org/covertype.test.arff.
//  - https://datasets.mlpack.org/covertype.test.labels.arff.

arma::mat dataset, testDataset;
arma::Row<size_t> labels, testLabels;
mlpack::data::DatasetInfo info;

mlpack::data::Load("covertype.train.arff", dataset, info, true);
mlpack::data::Load("covertype.train.labels.csv", labels, true);
mlpack::data::Load("covertype.test.arff", testDataset, info, true);
mlpack::data::Load("covertype.test.labels.csv", testLabels, true);

// Create a tree, and train on the training data.
mlpack::HoeffdingTree tree(info, 7 /* number of classes */, 0.98);
tree.MinSamples(500);
tree.CheckInterval(500);

tree.Train(dataset, labels);

// Print accuracy on the training and test set.
arma::Row<size_t> predictions, testPredictions;
tree.Classify(dataset, predictions);
tree.Classify(testDataset, testPredictions);

double trainAcc = (100.0 * arma::accu(predictions == labels)) / labels.n_elem;
double testAcc = (100.0 * arma::accu(testPredictions == testLabels)) /
    testLabels.n_elem;

std::cout << "When trained on the training data:" << std::endl;
std::cout << "  - Training set accuracy: " << trainAcc << "\%." << std::endl;
std::cout << "  - Test set accuracy:     " << testAcc << "\%." << std::endl;

// Now reset the tree, and train on the test set instead.
// The dataset info and number of classes has not changed, so we can just call
// Reset() with no arguments.
tree.Reset();
tree.Train(testDataset, testLabels);

// Print accuracy on the training and test set, now that we have trained on the
// test set.
tree.Classify(dataset, predictions);
tree.Classify(testDataset, testPredictions);

trainAcc = (100.0 * arma::accu(predictions == labels)) / labels.n_elem;
testAcc = (100.0 * arma::accu(testPredictions == testLabels)) /
    testLabels.n_elem;

std::cout << "When trained on the test data:" << std::endl;
std::cout << "  - Training set accuracy: " << trainAcc << "\%." << std::endl;
std::cout << "  - Test set accuracy:     " << testAcc << "\%." << std::endl;
```

---

### Advanced Functionality: Template Parameters

#### Using different element types.

`HoeffdingTree`'s constructors, `Train()`, and `Classify()` functions support
any data type, so long as it supports the Armadillo matrix API.  So, for
instance, learning can be done on single-precision floating-point data:

```c++
// 1000 random points in 10 dimensions.
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
mlpack::HoeffdingTree tree(dataset, labels, 5);

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

The `HoeffdingTree` class also supports several template parameters, which can
be used for custom behavior during learning.  The full signature of the class is
as follows:

```
HoeffdingTree<FitnessFunction,
              NumericSplitType,
              CategoricalSplitType>
```

 * `FitnessFunction`: the measure of goodness to use when deciding on tree
   splits
 * `NumericSplitType`: the strategy used for finding splits on numeric data
   dimensions
 * `CategoricalSplitType`: the strategy used for finding splits on categorical
   data dimensions

Below, details are given for the requirements of each of these template types.

---

#### `FitnessFunction`

 * Specifies the fitness function to use when learning a decision tree.
 * The `GiniImpurity` _(default)_ and `HoeffdingInformationGain` classes are
   available for drop-in usage.
 * `GiniImpurity` uses the [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity),
   which measures the probability of a randomly labeled random element in the
   split nodes being correctly labeled.
 * `HoeffdingInformationGain` uses the [information gain](https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain),
   which is based on the information-theoretic entropy of the possible splits.
 * A custom class must implement two functions:

```c++
// You can use this as a starting point for implementation.
class CustomFitnessFunction
{
  // Return the range (difference between maximum and minimum gain values).
  double Range(const size_t numClasses);

  // Compute the gain for the given split candidates represented in the matrix
  // `counts`.  `counts` is a matrix with `numChildren` columns and `numClasses`
  // rows, containing the number of points for each class held by each child.
  //
  // Note that the gain returned should be the gain for *all* child nodes (e.g.
  // all columns of `counts`).
  double Evaluate(const arma::Mat<size_t>& counts);
};
```

---

#### `NumericSplitType`

 * Specifies the strategy to be used during training when splitting a numeric
   feature.

 * Several options are already implemented and available for drop-in usage.
   - The `HoeffdingDoubleNumericSplit` _(default)_ class discretizes the given
     numeric data into a default of 10 bins.  This expects `double` to be the
     type of the input data.
   - The `HoeffdingFloatNumericSplit` class operates similarly to
     `HoeffdingDoubleNumericSplit`, but expects `float` to be the type of the
     input data.
   - The `BinaryNumericSplit` class splits numeric features in two in the way
     that maximizes gain.  This split type is more computationally expensive
     during training.

 * If a non-default `NumericSplitType` is specified, the following constructor
   forms can be used to pass constructed `NumericSplitType`s to the
   `HoeffdingTree` to use as copy-constructed templates during splitting:
    - `HoeffdingTree(dimensionality, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`
    - `HoeffdingTree(datasetInfo, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`
    - `HoeffdingTree(data, labels, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`
    - `HoeffdingTree(data, datasetInfo, labels, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`

 * A custom class must take a [`FitnessFunction`](#fitnessfunction) as a
   template parameter, implement several functions, and have an internal
   structure `SplitInfo` that is used at classification time:

```c++
// The job of this class is to track sufficient statistics of training data,
// returning gain information if a split were to happen according to this
// class's split strategy.
//
// For details, consult the HoeffdingNumericSplit and BinaryNumericSplit class
// implementations.
template<typename FitnessFunction>
class CustomNumericSplit
{
 public:
  // Create the split object with the given number of classes.
  CustomNumericSplit(const size_t numClasses);

  // Create the split from another split object with the given number of
  // classes.
  CustomNumericSplit(const size_t numClasses, const CustomNumericSplit& other);

  // Train on the given value with the given label.
  // Note that the type used here must match the element type of the training
  // data (so, e.g., if you plan to use `arma::fmat`, use `float` instead of
  // `double`).
  void Train(double value, const size_t label);

  // Given the points seen so far, evaluate the fitness function, returning the
  // gain if a split were to occur.  If this `NumericSplitType` class could
  // provide multiple possible splits, also return the second best fitness
  // value.  (If not, set secondBestFitness to 0.)
  void EvaluateFitnessFunction(double& bestFitness, double& secondBestFitness);

  // Return the number of children that would be created if a split were to
  // occur.  (For example, if this class implements a binary split, this should
  // return 2.)
  size_t NumChildren() const;

  // Given that a split should happen, return the majority classes of the
  // children and an initialized SplitInfo object.
  //
  // childMajorities should be set to have length equal to the number of
  // children that this strategy splits into, and the i'th element should be the
  // majority class label of the i'th child after splitting.
  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);

  // Return the current majority class of points seen so far.
  size_t MajorityClass() const;
  // Return the probability of the majority class given the points seen so far.
  double MajorityProbability() const;

  // Serialize (load/save) the split object using cereal.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

  // The SplitInfo class should implement two functions.  It is used at
  // prediction time, after a split has occurred, and should contain the
  // information necessary to classify a point.
  //
  // The SplitInfo class must implement two methods; one for classification and
  // one for serialization.
  class SplitInfo
  {
   public:
    // Given that the point in the split dimension has the value `value`, return
    // the index of the child that the traversal should go to.
    template<typename eT>
    size_t CalculateDirection(const eT& value) const;

    // Serialize the split (load/save) using cereal.
    template<typename Archive>
    void serialize(Archive& ar, const uint32_t version);
  };
};
```

---

#### `CategoricalSplitType`

 * Specifies the strategy to be used during training when splitting a
    categorical feature.

 * The `HoeffdingCategoricalSplit` _(default)_ is available for drop-in usage
   and splits all categories into their own node.

 * If a non-default `CategoricalSplitType` is specified, the following
   constructor forms can be used to pass constructed `CategoricalSplitType`s to
   the `HoeffdingTree` to use as copy-constructed templates during splitting:
    - `HoeffdingTree(dimensionality, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`
    - `HoeffdingTree(datasetInfo, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`
    - `HoeffdingTree(data, labels, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`
    - `HoeffdingTree(data, datasetInfo, labels, numClasses, successProbability, maxSamples, checkInterval, minSamples, categoricalSplit, numericSplit)`

 * A custom class must take a [`FitnessFunction`](#fitnessfunction) as a
   template parameter, implement several functions, and have an internal
   structure `SplitInfo` that is used at classification time:

```c++
// The job of this class is to track sufficient statistics of training data,
// returning gain information if a split were to happen according to this
// class's split strategy.
//
// For details, consult the HoeffdingCategoricalSplit class implementation.
template<typename FitnessFunction>
class CustomCategoricalSplit
{
 public:
  // Create the split object with the given number of classes.  The dimension
  // that this object tracks has `numCategories` possible category values.
  CustomCategoricalSplit(const size_t numCategories, const size_t numClasses);

  // Create the split object from another split object with the given number of
  // classes.  The dimension that this object tracks has `numCategories`
  // possible category values.
  CustomCategoricalSplit(const size_t numCategories, const size_t numClasses,
                         const CustomCategoricalSplit& other);

  // Train on the given value with the given label.
  // Note that the type used here must match the element type of the training
  // data (so, e.g., if you plan to use `arma::fmat`, use `float` instead of
  // `double`).
  void Train(double value, const size_t label);

  // Given the points seen so far, evaluate the fitness function, returning the
  // gain if a split were to occur.  If this `NumericSplitType` class could
  // provide multiple possible splits, also return the second best fitness
  // value.  (If not, set secondBestFitness to 0.)
  void EvaluateFitnessFunction(double& bestFitness, double& secondBestFitness);

  // Return the number of children that would be created if a split were to
  // occur.  (For example, if this class implements a binary split, this should
  // return 2.)
  size_t NumChildren() const;

  // Given that a split should happen, return the majority classes of the
  // children and an initialized SplitInfo object.
  //
  // childMajorities should be set to have length equal to the number of
  // children that this strategy splits into, and the i'th element should be the
  // majority class label of the i'th child after splitting.
  void Split(arma::Col<size_t>& childMajorities, SplitInfo& splitInfo);

  // Return the current majority class of points seen so far.
  size_t MajorityClass() const;
  // Return the probability of the majority class given the points seen so far.
  double MajorityProbability() const;

  // Serialize (load/save) the split object using cereal.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

  // The SplitInfo class should implement two functions.  It is used at
  // prediction time, after a split has occurred, and should contain the
  // information necessary to classify a point.
  //
  // The SplitInfo class must implement two methods; one for classification and
  // one for serialization.
  class SplitInfo
  {
   public:
    // Given that the point in the split dimension has the value `value`, return
    // the index of the child that the traversal should go to.
    template<typename eT>
    size_t CalculateDirection(const eT& value) const;

    // Serialize the split (load/save) using cereal.
    template<typename Archive>
    void serialize(Archive& ar, const uint32_t version);
  };
};
```
