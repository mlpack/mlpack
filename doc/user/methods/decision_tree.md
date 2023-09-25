## `DecisionTree`

The `DecisionTree` class implements a decision tree classifier that supports
numerical and categorical features, by default using Gini gain to choose which
feature to split on.  The class offers several template parameters and several
constructor parameters that can be used to control the behavior of the tree.

*Basic usage excerpt:*

```c++
DecisionTree tree(3); // [Step 1](#constructors): construct object.
tree.Train(data, labels, 3); // [Step 2](#training): train model.
tree.Classify(test_data, test_predictions); // [Step 3](#classification): use model to classify points.
```

*Quick links:*

 * [Constructors](#constructors): create `DecisionTree` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model).
 * [Other functionality](#other_functionality) for loading, saving, and
   inspecting.
 * [Template parameters](#template_parameters) for custom behavior.
 * [Examples](#examples) of simple usage and links to detailed example projects.

*See also*:

 * [Random forests](#random_forests) <!-- TODO: fix link! -->
 * [mlpack classifiers](#mlpack_classifiers) <!-- TODO: fix link! -->
 * [Decision tree on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
 * [Decision tree learning on Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

### Constructors

Construct a `DecisionTree` object using one of the constructors below.

*Forms*:

 * `DecisionTree()`
 * `DecisionTree(numClasses)`
   - Initialize tree without training.
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Classify()`](#classify).


 * `DecisionTree(data, labels, numClasses)`
 * `DecisionTree(data, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical-only data.
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).


 * `DecisionTree(data, datasetInfo, labels, numClasses)`
 * `DecisionTree(data, datasetInfo, labels, numClasses, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed categorical data.
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).


<!-- TODO: weighted numerical-only constructors -->

 * `DecisionTree(data, datasetInfo, labels, numClasses, weights)`
 * `DecisionTree(data, datasetInfo, labels, numClasses, weights, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on weighted mixed categorical data.
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).
   - `weights` should be a vector of length `data.n_cols`, containing instance
     weights for each point in `data`.

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
| `labels` | [`arma::Row<size_t>`]('../matrices.md') | Training labels, between `0` and `numClasses - 1` (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`]('../matrices.md') | Weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
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

***Note***: training is not incremental.  A second call to `Train()` will
retrain the decision tree from scratch.

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

See also the following fully-working examples:

 - [Loan default prediction with `DecisionTree`](https://github.com/mlpack/examples/blob/master/loan_default_prediction_with_decision_tree/loan-default-prediction-with-decision-tree-cpp.ipynb)

### Advanced Functionality: Template Parameters

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

 * `FitnessFunction`
    - Specifies the fitness function to use when learning a decision tree.
    - The `GiniGain` and `InformationGain` classes are available for drop-in
      usage.
    - A custom class must implement three functions:
        * `double Range(const size_t numClasses)`: return the range (difference
          between maximum and minimum gain values).
        * `double Evaluate(RowType labels, size_t numClasses, WeightVecType
          weights)`: compute the gain for the given vector of labels with
          associated instance weights.
        * `double EvaluatePtr(CountType* counts, size_t numClasses, CountType
          totalCount)`: compute the gain for the counted set of labels where
          `counts[i]` contains the number of points with label `i`, with total
          set size `totalCount`.
