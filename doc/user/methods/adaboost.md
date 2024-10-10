## `AdaBoost`

The `AdaBoost` class implements the 'adaptive boosting' classifier AdaBoost.MH.
This classifier is an ensemble of weak learners.  The `AdaBoost` class offers
control over the weak learners and other behavior via template parameters.  By
default, the `Perceptron` class is used as a weak learner.

`AdaBoost` is useful for classifying points with _discrete labels_ (i.e. `0`,
`1`, `2`).

#### Simple usage example:

Train an AdaBoost model on random data and predict labels on a random test set.

```c++
// Train an AdaBoost model on random data and predict labels on test data:

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::AdaBoost ab;                   // Step 1: create model.
ab.Train(dataset, labels, 5);          // Step 2: train model.
arma::Row<size_t> predictions;
ab.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `AdaBoost` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.
 * [Advanced template examples](#advanced-functionality-examples) of use with
   custom template parameters.

#### See also:

 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [`Perceptron`](perceptron.md)
 * [`DecisionTree`](decision_tree.md)
 * [AdaBoost on Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
 * [AdaBoost.MH paper (pdf)](https://dl.acm.org/doi/pdf/10.1145/279943.279960)

### Constructors

 * `ab = AdaBoost(tolerance=1e-6)`
   - Initialize model without training.
   - You will need to call [`Train()`](#training) later to train the tree before
     calling [`Classify()`](#classification).

---

 * `ab = AdaBoost(data, labels, numClasses, maxIterations=100, tolerance=1e-6)`
   - Train model using default weak learner hyperparameters.

---

 * `ab = AdaBoost(data, labels, numClasses, maxIterations=100, tolerance=1e-6, _[weak learner hyperparameters...]_)`
   - Train model with custom weak learner hyperparameters.
   - Hyperparameters for the weak learner are any arguments to the weak
     learner's `Train()` function that come after `numClasses` or `weights`.
   - The only hyperparameter for the default weak learner (`Perceptron`) is
     `maxIterations`.
   - See [examples of this constructor in use](#simple-examples).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `weakLearner` | `Perceptron` | An initialized weak learner whose hyperparameters will be used as settings for weak learners during training. | _(N/A)_ |
| `maxIterations` | `size_t` | Maximum number of iterations of AdaBoost.MH to use.  This is the maximum number of weak learners to train.  (0 means no limit, and weak learners will be trained until the tolerance is met.) | `100` |
| `tolerance` | `double` | When the weighted residual (`r_t`) of the model goes below `tolerance`, training will terminate and no more weak learners will be added. | `1e-6` |

As an alternative to passing hyperparameters, each hyperparameter can be set
with a standalone method.  The following functions can be used before calling
`Train()` to set hyperparameters:

 * `ab.MaxIterations() = maxIter;` will set the maximum number of weak learners
   during training to `maxIter`.
 * `ab.Tolerance() = tol;` will set the tolerance to `tol`.

***Note:*** different types of weak learners can be used than
[`Perceptron`](perceptron.md), by changing the [`WeakLearnerType` template
parameter](#advanced-functionality-template-parameters).

### Training

If training is not done as part of the constructor call, it can be done with one
of the versions of the `Train()` member function.  For an instance of `AdaBoost`
named `ab`, the following functions for training are available:

 * `ab.Train(data, labels, numClasses, maxIterations=100, tolerance=1e-6)`
   - Train model using default weak learner parameters.

---

 * `ab.Train(data, labels, numClasses, maxIterations=100, tolerance=1e-6, [weak learner hyperparameters...])`
   - Train model with custom weak learner parameters.
   - Hyperparameters for the weak learner are any arguments to the weak
     learner's `Train()` function that come after `numClasses` or `weights`.
   - The only hyperparameter for the default weak learner (`Perceptron`) is
     `maxIterations`.
   - See [examples of this form in use](#simple-examples).

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes***:

 * Training is not incremental.  A second call to `Train()` will retrain the
   AdaBoost model from scratch.

 * `Train()` returns a `double` indicating an upper bound on the training error
   (specifically, the product of _Zt_ values, as described in the paper).

### Classification

Once an `AdaBoost` model is trained, the `Classify()` member function can be
used to make class predictions for new data.

 * `size_t predictedClass = ab.Classify(point)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class.

---

 * `ab.Classify(point, prediction, probabilitiesVec)`
   - ***(Single-point)***
   - Classify a single point and compute class probabilities.
   - The predicted class is stored in `prediction`.
   - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `ab.Classify(data, predictions)`
   - ***(Multi-point)***
   - Classify a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `ab.Classify(data, predictions, probabilities)`
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
| _single-point_ | `probabilitiesVec` | [`arma::vec&`](../matrices.md) | `arma::vec&` to store class probabilities into. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into; will be set to length `data.n_cols`. |
| _multi-point_ | `probabilities` | [`arma::mat&`](../matrices.md) | Matrix to store class probabilities into (number of rows will be equal to number of classes; number of columns will be equal to `data.n_cols`). |

### Other Functionality

 * An `AdaBoost` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

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

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `AdaBoost` class.

---

Train an AdaBoost model using the hyperparameters from an existing weak learner.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("iris.labels.csv", labels, true);

mlpack::AdaBoost ab;
// Train with a custom number of perceptron iterations, and custom AdaBoost
// parameters.
ab.Train(dataset, labels, 3, 75 /* maximum number of weak learners */,
                             1e-6 /* tolerance for AdaBoost convergence */,
                             100 /* maximum number of perceptron iterations */);

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
mlpack::data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("iris.labels.csv", labels, true);

mlpack::AdaBoost ab;
ab.MaxIterations() = 50; // Use at most 50 weak learners.
ab.Tolerance() = 1e-4; // Set a custom tolerance for convergence.

// Now train, using the hyperparameters specified above.
ab.Train(dataset, labels, 3);

// Save the model to `adaboost_model.bin`.
mlpack::data::Save("adaboost_model.bin", "adaboost_model", ab, true);
```

---

Load an AdaBoost model and print some information about it.

```c++
// Load a saved model named "adaboost_model" from `adaboost_model.bin`.
mlpack::AdaBoost ab;
mlpack::data::Load("adaboost_model.bin", "adaboost_model", ab, true);

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

 - [Graduate admission classification with `AdaBoost`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/adaboost/graduate_admission_classification/graduate-admission-classification-cpp.ipynb)

### Advanced Functionality: Template Parameters

The `AdaBoost` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```
AdaBoost<WeakLearnerType, MatType>
```

 * `WeakLearnerType`: the weak classifier to ensemble in the AdaBoost model.
 * `MatType`: specifies the type of matrix used for learning and internal
   representation of model parameters.

---

#### `WeakLearnerType`

 * Specifies the weak learner to use when constructing an AdaBoost model.
 * The default `WeakLearnerType` is [`Perceptron<>`](perceptron.md).
 * The `ID3DecisionStump` class (a custom variant of
   [`DecisionTree`](decision_tree.md)) is available for drop-in usage as a weak
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

---

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
// Note that we specify decision stumps as the weak learner type, and pass
// hyperparameters for the decision stump (these could be omitted).  See the
// DecisionTree documentation for more details on the ID3DecisionStump-specific
// hyperparameters.
mlpack::AdaBoost<mlpack::ID3DecisionStump> ab(dataset, labels, 5,
    25 /* maximum number of decision stumps */,
    1e-6 /* tolerance for convergence of AdaBoost */,
    /** Hyperparameters specific to ID3DecisionStump: **/
    10 /* minimum number of points in each leaf of the decision stump */,
    1e-5 /* minimum gain for splitting the root node of the decision stump */);

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
// The weak learner type is now a floating-point Perceptron.
typedef mlpack::Perceptron<mlpack::SimpleWeightUpdate,
                           mlpack::ZeroInitialization,
                           arma::fmat> PerceptronType;
mlpack::AdaBoost<PerceptronType, arma::fmat> ab(dataset, labels, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
ab.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```
