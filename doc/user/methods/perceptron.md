## `Perceptron`

The `Perceptron` class implements the simple perceptron classifier originally
implemented by Frank Rosenblatt in 1958.  The perceptron is a linear classifier,
and can be understood as a trivial neural network with one neuron that uses the
step function as an activation function.  mlpack's implementation of the
`Perceptron` class also offers several template parameters that can be used to
control the behavior of the perceptron.

Perceptrons are useful for classifying points with _discrete labels_ (i.e., `0`,
`1`, `2`).  Because they are simple classifiers, they are also useful as _weak
learners_ for the [`AdaBoost`](adaboost.md) boosting classifier.

#### Simple usage example:

```c++
// Train a perceptron on random numeric data and predict labels on test data:

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::Perceptron p;                 // Step 1: create model.
p.Train(dataset, labels, 5);          // Step 2: train model.
arma::Row<size_t> predictions;
p.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 1) << " test points classified as class "
    << "1." << std::endl;
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
 * [Advanced template examples](#advanced-functionality-examples) of use with
   custom template parameters.

#### See also:

 * [`NaiveBayesClassifier`](naive_bayes_classifier.md), another simple classifier
 * [`AdaBoost`](adaboost.md)
 * [`FFN`](/src/mlpack/methods/ann/ffn.hpp)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [Perceptron on Wikipedia](https://en.wikipedia.org/wiki/Perceptron)

### Constructors

Construct a `Perceptron` object using one of the constructors below.  Defaults
and types are detailed in the [Constructor Parameters](#constructor-parameters)
section below.

#### Forms:

 * `p = Perceptron()`
   - Initialize perceptron without training.
   - You will need to call [`Train()`](#training) later to train the perceptron
     before calling [`Classify()`](#classification).

---

 * `p = Perceptron(numClasses, dimensionality, maxIterations=1000)`
   - Initialize perceptron with all-zero weights and biases.
   - `Classify()` can immediately be used; training is not required with this
     form.

---

 * `p = Perceptron(data, labels, numClasses,          maxIterations=1000)`
 * `p = Perceptron(data, labels, numClasses, weights, maxIterations=1000)`
   - Train the perceptron (optionally with instance weights).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](../load_save.md#mixed-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, between [`0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`](../matrices.md) | Weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `dimensionality` | `size_t` | Dimensionality of data (only used if an initialized but untrained model is desired). | _(N/A)_ |
| `maxIterations` | `size_t` | Maximum number of iterations during training.  Can also be set with `MaxIterations()`. | `1000` |

As an alternative to passing `maxIterations`, it can be set with a standalone
method.  The following function can be used before calling `Train()` to set
the maximum number of iterations:

 * `p.MaxIterations() = maxIter;` will set the maximum number of iterations
   during training to `maxIter`.

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `p.Train(data, labels, numClasses, maxIterations=1000)`
   - Train the perceptron on unweighted data.

---

 * `p.Train(data, labels, numClasses, weights, maxIterations=1000)`
   - Train the perceptron on data with instance weights.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes***:

 * Training is incremental.  Successive calls to `Train()` will not reinitialize
   the model, unless the given data has different dimensionality or `numClasses`
   is different.  To reinitialize the model, call `Reset()` (see
   [Other Functionality](#other-functionality)).

 * If `maxIterations` is not passed, but has been set in the constructor or with
   `MaxIterations()`, the previous setting will be used.

### Classification

Once a `Perceptron` is trained, the `Classify()` member function can be used to
make class predictions for new data.

 * `size_t predictedClass = p.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `p.Classify(data, predictions)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The prediction for data point `i` can be accessed with `predictions[i]`.

---

***Note***: perceptrons do not provide any measure resembling probabilities
during classification, and thus a version of `Classify()` that computes class
probabilities is not available.

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into.  Will be set to length `data.n_cols`. |

### Other Functionality

 * A `Perceptron` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `p.NumClasses()` will return a `size_t` indicating the number of classes the
   perceptron was trained on.

 * `p.Biases()` will return an `arma::vec` with the biases of the model (each
   element corresponds to the bias for a class).

 * `p.Weights()` will return an `arma::mat` with the weights of the model (each
   column corresponds to the weights for one class label).

 * `p.Reset()` will re-initialize the weights and biases of the model.

For complete functionality, the [source
code](/src/mlpack/methods/perceptron/perceptron.hpp) can be consulted.  Each
method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`Perceptron`.

---

Train a perceptron multiple times, incrementally, with custom hyperparameters,
and save the resulting model to disk.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("iris.labels.csv", labels, true);

// Create a Perceptron object.
mlpack::Perceptron p;
// Set the maximum number of iterations to 100.  (This can also be done in the
// constructor.)
p.MaxIterations() = 100;

// Train the model for up to 100 iterations.
p.Train(dataset, labels, 3);

// Now, compute and print accuracy on the training set.
arma::Row<size_t> predictions;
p.Classify(dataset, predictions);
std::cout << "Training set accuracy after 100 iterations: "
    << (100.0 * double(arma::accu(labels == predictions)) / labels.n_elem)
    << "\%." << std::endl;

// Train for another 250 iterations and compute training set accuracy again.
p.Train(dataset, labels, 3, 250);
p.Classify(dataset, predictions);
std::cout << "Training set accuracy after 350 iterations: "
    << (100.0 * double(arma::accu(labels == predictions)) / labels.n_elem)
    << "\%." << std::endl;

// Save the perceptron to disk for later use.
mlpack::data::Save("perceptron.bin", "perceptron", p);
```

---

Load a saved perceptron from disk and print information about it.

```c++
mlpack::Perceptron p;
// This call assumes a perceptron called "p" has already been saved to
// `perceptron.bin` with `data::Save()`.
mlpack::data::Load("perceptron.bin", "p", p, true);

if (p.NumClasses() > 0)
{
  std::cout << "The perceptron in `perceptron.bin` was trained on "
      << p.NumClasses() << " classes." << std::endl;
  std::cout << "The dimensionality of the perceptron model is "
      << p.Weights().n_rows << "." << std::endl;
  std::cout << "The bias weights for each class are:" << std::endl;
  for (size_t i = 0; i < p.NumClasses(); ++i)
    std::cout << "  - Class " << i << ": " << p.Biases()[i] << std::endl;
}
else
{
  std::cout << "The perceptron in `perceptron.bin` has not been trained."
      << std::endl;
}
```

---

### Advanced Functionality: Template Parameters

The `Perceptron` class also supports several template parameters, which can be
used for custom behavior.  The full signature of the class is as follows:

```
Perceptron<LearnPolicy,
           WeightInitializationPolicy,
           MatType>
```

 * `LearnPolicy`: the strategy used to learn the weights during training.
 * `WeightInitializationPolicy`: the way that weights are initialized before
   training.
 * `MatType`: specifies the type of matrix used for learning and internal
   representation of weights and biases.

---

#### `LearnPolicy`

 * Specifies the step to be taken when a point is misclassified.
 * The `SimpleWeightUpdate` class is available, and is the default.
 * A custom class must implement only one function:

```c++
// You can use this as a starting point for implementation.
class CustomLearnPolicy
{
  // Update the weights and biases in the `weights` matrix and the `biases`
  // vector given that the model currently classified `trainingPoint` as having
  // the label `incorrectClass`, when in reality it has the label
  // `correctClass`.  If `instanceWeight` is given, it specifies the instance
  // weight for the given `trainingPoint`.
  //
  // `VecType` will be an Armadillo-like vector type.  It will be a column from
  // the training data matrix (`data`) given to `Train()` or to the constructor.
  //
  // `eT` is the element type of the Perceptron (e.g. `float`, `double`).
  template<typename VecType, typename eT>
  void UpdateWeights(const VecType& trainingPoint,
                     arma::Mat<eT>& weights,
                     arma::Col<eT>& biases,
                     const size_t incorrectClass,
                     const size_t correctClass,
                     const double instanceWeight = 1.0);
};
```

---

#### `WeightInitializationPolicy`

 * Specifies how the weights matrix and biases vector should be initialized when
   the `Perceptron` object is created, or when `Reset()` is called.
 * The `ZeroInitialization` _(default)_ and `RandomPerceptronInitialization`
   classes are available for drop-in usage.
 * `RandomPerceptronInitialization` will initialize weights and biases using a
   uniform random distribution between 0 and 1.
 * A custom class must implement only one function:

```c++
// You can use this as a starting point for implementation.
class CustomWeightInitializationPolicy
{
  // Initialize the `weights` matrix and `biases` vector, given that the model
  // will have dimensionality of `numFeatures` (that is, the training data
  // matrix will have `numFeatures` rows), and the training data has
  // `numClasses` classes.
  //
  // The initialized `weights` matrix should have `numFeatures` rows and
  // `numClasses` columns, and the initialized `biases` vector should have
  // `numClasses` elements.
  //
  // `eT` specifies the element type of the weights and biases; it may be
  // `double`, `float`, or another floating-point type.
  template<typename eT>
  inline static void Initialize(arma::Mat<eT>& weights,
                                arma::Col<eT>& biases,
                                const size_t numFeatures,
                                const size_t numClasses)
  {
    weights.randu(numFeatures, numClasses);
    biases.randu(numClasses);
  }
};
```

---

#### `MatType`

 * Specifies the matrix type to use for data when learning a perceptron.
 * By default, `MatType` is `arma::mat` (dense 64-bit precision matrix).
 * Any matrix type implementing the Armadillo API will work; so, for instance,
   `arma::fmat` or `arma::sp_mat` can be used.

### Advanced Functionality Examples

Train a `Perceptron` with random initialization, instead of zero initialization
of weights.

```c++
// 1000 random points in 10 dimensions.
arma::mat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.  Weights will be initialized randomly.
mlpack::Perceptron<mlpack::SimpleWeightUpdate,
                   mlpack::RandomPerceptronInitialization> p(
    dataset, labels, 5);

// Create test data (500 points).
arma::mat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
p.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 1) << " test points classified as class "
    << "1." << std::endl;
```

---

Train a `Perceptron` on sparse 32-bit floating point data.

```c++

// 1000 sparse random points in 100 dimensions, with 1% nonzero elements.
arma::sp_fmat dataset;
dataset.sprandu(100, 1000, 0.01);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
mlpack::Perceptron p(dataset, labels, 5);

// Create test data (500 points).
arma::sp_fmat testDataset;
testDataset.sprandu(100, 500, 0.01);
arma::Row<size_t> predictions;
p.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 1) << " test points classified as class "
    << "1." << std::endl;
```

---
