## `AdaBoost`

The `AdaBoost` class implements the 'adaptive boosting' classifier AdaBoost.MH.
This classifier is an ensemble of weak learners.  The `AdaBoost` class offers
control over the weak learners and other behavior via template parameters.  By
default, the `Perceptron` class is used as a weak learner.

`AdaBoost` is useful for classifying points with _discrete labels_ (i.e. `0`,
`1`, `2`).

#### Basic usage example excerpt:

```c++
AdaBoost ab;                            // Step 1: construct object.
ab.Train(data, labels, 3);              // Step 2: train model.
ab.Classify(testData, testPredictions); // Step 3: use model to classify.
```

#### Quick links:

 * [Constructors](#constructors): create `AdaBoost` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classify): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.
 * [Advanced template examples](#advanced-functionality-examples) of use with
   custom template parameters.

#### See also:

 * [mlpack classifiers](#mlpack_classifiers) <!-- TODO: fix link! -->
 * [`Perceptron`](#perceptron) <!-- TODO: fix link! -->
 * [`DecisionTree`](#decision_tree) <!-- TODO: fix link! -->
 * [AdaBoost on Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
 * [AdaBoost.MH paper (pdf)](https://dl.acm.org/doi/pdf/10.1145/279943.279960)

### Constructors

Construct an `AdaBoost` object using one of the constructors below.  Defaults
and types are detailed in the [Constructor Parameters](#constructor-parameters)
section below.

#### Forms:

 * `AdaBoost()`
 * `AdaBoost(tolerance)`
   - **Initialize model without training.**
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Classify()`](#classification).

---
<!-- TODO: add this variant! -->

 * `AdaBoost(data, labels, numClasses)`
 * `AdaBoost(data, labels, numClasses, maxIterations, tolerance)`
   - **Train model using default weak learner parameters.**
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).

---

 * `AdaBoost(data, labels, numClasses, weakLearner)`
 * `AdaBoost(data, labels, numClasses, weakLearner, maxIterations, tolerance)`
   - **Train model with custom weak learner parameters.**
   - The given `weakLearner` does not need to be trained; any hyperparameter
     settings in `weakLearner` are used for training each AdaBoost weak
     learner (see the [simple examples](#simple-examples)). <!-- TODO: link to
specific example -->
   - If hyperparameters are not specified, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).

---

<!-- TODO: a variant that allows passing hyperparameters directly! -->

---

#### Constructor Parameters:

<!-- TODOs for table below:
    * better link for column-major matrices
    * update matrices.md to include a section of labels and NormalizeLabels()
-->

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`]('../matrices.md') | Training labels, between `0` and `numClasses - 1` (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `weakLearner` | `Perceptron` | An initialized weak learner whose
hyperparameters will be used as settings for weak learners during training. |
_(N/A)_ |
| `maxIterations` | `size_t` | Maximum number of iterations of AdaBoost.MH to use.  This is the maximum number of weak learners to train.  (0 means no limit, and weak learners will be trained until the tolerance is met.) | `100` |
| `tolerance` | `double` | When the weighted residual (`r_t`) of the model goes
below `tolerance`, training will terminate and no more weak learners will be
added. | `1e-6` |

As an alternative to passing hyperparameters, each hyperparameter can be set
with a standalone method.  For an instance of `AdaBoost` named `ab`, the
following functions can be used before calling `Train()` to set hyperparameters:

<!-- TODO: actually fix this in code -->

 * `ab.MaxIterations() = maxIter;` will set the maximum number of weak learners
   during training to `maxIter`.
 * `ab.Tolerance() = tol;` will set the tolerance to `tol`.

<!-- TODO: fix links -->

***Note:*** different types of weak learners can be used than
[`Perceptron`](#perceptron), by changing the [`WeakLearnerType` template
parameter](#advanced-functionality-template-parameters).

### Training

If training is not done as part of the constructor call, it can be done with one
of the versions of the `Train()` member function.  For an instance of `AdaBoost`
named `ab`, the following functions for training are available:

<!-- TODO: implement this variant -->

 * `ab.Train(data, labels, numClasses)`
 * `ab.Train(data, labels, numClasses, maxIterations, tolerance)`
   - **Train model using default weak learner parameters.**
   - If hyperparameters are not specified here, and have not been otherwise set,
     default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).

---

 * `ab.Train(data, labels, numClasses, weakLearner)`
 * `ab.Train(data, labels, numClasses, weakLearner, maxIterations, tolerance)`
   - **Train model with custom weak learner parameters.**
   - The given `weakLearner` does not need to be trained; any hyperparameter
     settings in `weakLearner` are used for training each AdaBoost weak learner
     (see the [simple examples](#simple-examples)). <!-- TODO: link to specific
example -->
   - If hyperparameters for AdaBoost are not specified, and have not been
     otherwise set, default values are used.
   - `labels` should be a vector of length `data.n_cols`, containing values from
     `0` to `numClasses - 1` (inclusive).

---

<!-- TODO: a variant that allows passing hyperparameters directly! -->

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Note***: training is not incremental.  A second call to `Train()` will
retrain the AdaBoost model from scratch.

### Classification

Once an `AdaBoost` model is trained, the `Classify()` member function can be
used to make class predictions for new data.  Defaults and types are detailed in
the [Classification Parameters](#classification-parameters) section below.

#### Forms:

<!-- TODO: add single-point forms -->

 * `size_t predictedClass = ab.Classify(point)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class.

---

 * `ab.Classify(point, prediction, probabilities_vec)`
   - ***(Single-point)***
   - Classify a single point and compute class probabilities.
   - The predicted class is stored in `prediction`.
   - The class probabilities are stored in `probabilities_vec`, which is set to
     length `numClasses`.
   - The probability of class `i` can be accessed with `probabilities_vec[i]`.

---

 * `ab.Classify(data, predictions)`
   - ***(Multi-point)***
   - Classify a set of points.
   - The predicted class of each point is stored in `predictions`, which is set
     to length `data.n_cols`.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `ab.Classify(data, predictions, probabilities)`
   - ***(Multi-point)***
   - Classify a set of points and compute class probabilities for each point.
   - The predicted class of each point is stored in `predictions`, which is set
     to length `data.n_cols`.
   - The prediction for data point `i` can be accessed with `predictions[i]`.
   - The class probabilities for each point are stored in `probabilities`, which
     is set to size `numClasses` by `data.n_cols`.
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

### Other Functionality

<!-- TODO: we should point directly to the documentation of those functions -->

 * An `AdaBoost` model can be serialized with [`data::Save()`](../formats.md)
   and [`data::Load()`](../formats.md).

 * `ab.NumClasses()` will return a `size_t` indicating the number of classes the
   model was trained on.

 * `ab.WeakLearners()` will return a `size_t` indicating the number of weak
   learners that the model currently contains.

 * `ab.Alpha(i)` will return the weight of weak learner `i`.

 * `ab.WeakLearner(i)` will return the `i`th weak learner.

For complete functionality, the [source
code](/src/mlpack/methods/adaboost/adaboost.hpp) can be consulted.  Each method
is fully documented.

### Simple Examples

Train an AdaBoost model on random data and predict labels on a random test set.

```c++
// 1000 random points in 10 dimensions.
arma::mat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
AdaBoost<> ab(dataset, labels, 5);

// Create test data (500 points).
arma::mat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
ab.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```

---

Train an AdaBoost model using the hyperparameters from an existing weak learner.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
data::Load("iris.labels.csv", dataset, true);

// Create a weak learner with the desired hyperparameters.
Perceptron<> p;
p.MaxIterations() = 500; // We'll use a custom maximum number of iterations.

AdaBoost<> ab;
ab.Train(dataset, labels, 3, p);

// Now predict the label of a point and the probabilities of each class.
size_t prediction;
arma::rowvec probabilities;
ab.Classify(dataset.col(10), prediction, probabilities);

std::cout << "Point 11 is predicted to have class " << prediction << "."
    << std::endl;
std::cout << "Probabilities of each class: " << probabilities;
```

---

Before training an AdaBoost model, set hyperparameters individually.  Save the
trained model to disk.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
data::Load("iris_labels.csv", dataset, true);

AdaBoost<> ab;
ab.MaxIterations() = 50; // Use at most 50 weak learners.
ab.Tolerance() = 1e-4; // Set a custom tolerance for convergence.

// Now train, using the hyperparameters specified above.
ab.Train(dataset, labels, 3);

// Save the model to `adaboost_model.bin`.
data::Save("adaboost_model.bin", "adaboost_model", ab, true);
```

---

Load an AdaBoost model and print some information about it.

```c++
// Load a saved model named "adaboost_model" from `adaboost_model.bin`.
AdaBoost<> ab;
data::Load("adaboost_model.bin", "adaboost_model", ab, true);

std::cout << "Details about the model in `adaboost_model.bin`:" << std::endl;
std::cout << "  - Trained on " << ab.NumClasses() << " classes." << std::endl;
std::cout << "  - Tolerance used for training: " << ab.Tolerance() << "."
    << std::endl;
std::cout << "  - Number of perceptron weak learners in model"
    << ab.WeakLearners() << "." << std::endl;

// Print some details about the first weak learner, if available.  The weak
// learner type is `Perceptron<>`.
if (ab.WeakLearners() > 0)
{
  std::cout << "  - Weight of first perceptron weak learner: " << ab.Alpha(0)
      << "." << std::endl;
  std::cout << "  - Biases of first perceptron learner: "
      << ab.WeakLearner(0).Biases().t();
}
```

---

See also the following fully-working examples:

 - [Graduate admission classification with `AdaBoost`](https://github.com/mlpack/examples/blob/master/graduate_admission_classification_with_Adaboost/graduate-admission-classification-with-adaboost-cpp.ipynb)

### Advanced Functionality: Template Parameters

The `AdaBoost` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```c++
AdaBoost<WeakLearnerType, MatType>
```

#### `WeakLearnerType`

<!-- TODO: fix links! -->

 * Specifies the weak learner to use when constructing an AdaBoost model.
 * The default `WeakLearnerType` is [`Perceptron<>`](#perceptron).
 * The `ID3DecisionStump` class (a custom variant of
   [`DecisionTree`](#decision_tree)) is available for drop-in usage as a weak
   learner.
 * Any custom variant of `Perceptron<>` or `DecisionTree<>` can be used; e.g.,
   `Perceptron<SimpleWeightUpdate, RandomPerceptronInitialization>`.
 * A custom class must implement the following functions for training and
   classification.  Note that this is the same API as mlpack's classifiers that
   support instance weights for learning, and so any mlpack classifier
   supporting instance weights can also be used.

```c++
// You can use this as a starting point for implementation.
class CustomWeakLearner
{
 public:
  // Train the model with the given hyperparameters.
  //
  //  * `MatType` will be an Armadillo-like matrix type (typically `arma::mat`);
  //    this is the same type as the `MatType` template parameter for the
  //    `AdaBoost` class.
  //
  //  * `data` and `labels` are the same dataset passed to the `Train()` method
  //    of `AdaBoost`.
  //
  //  * `weights` contains instance weights for each point (column) of `data`.
  //
  // Note: there is no restriction on the number or types of hyperparameters
  // that can be used, but, they do need default arguments.  The example here
  // includes two.
  template<typename MatType>
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const arma::rowvec& weights,
             const size_t hyperparameterA = 10,
             const double hyperparameterB = 0.1);

  // Classify the given point.  `VecType` will be an Armadillo-like type that is
  // a vector that represents a single point.
  template<typename VecType>
  size_t Classify(const VecType& point);

  // Classify the given points in `data`, storing the predicted classifications
  // in `predictions`.
  template<typename MatType>
  void Classify(const MatType& data, arma::Row<size_t>& predictions);
};
```

#### `MatType`

 * Specifies the matrix type to use for data when learning a model (or
   predicting with one).
 * By default, `MatType` is `arma::mat` (dense 64-bit precision matrix).
 * Any matrix type implementing the Armadillo API will work; so, for instance,
   `arma::fmat` or `arma::sp_mat` can also be used.

### Advanced Functionality Examples

Train an AdaBoost model using decision stumps as the weak learner (use a
different `WeakLearnerType`).

```c++
// 1000 random points in 10 dimensions.
arma::mat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
// Note that we specify decision stumps as the weak learner type.
AdaBoost<ID3DecisionStump> ab(dataset, labels, 5);

// Create test data (500 points).
arma::mat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
ab.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```

---

Train an AdaBoost model on 32-bit floating-point precision data (use a different
`MatType`).

```c++
// 1000 random points in 10 dimensions, using 32-bit precision (float).
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor, using floating-point data.
// (TODO: do we have to explicitly write MatType?)
AdaBoost<> ab(dataset, labels, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
ab.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```
