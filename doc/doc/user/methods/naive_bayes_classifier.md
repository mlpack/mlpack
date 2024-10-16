## `NaiveBayesClassifier`

The `NaiveBayesClassifier` implements a trivial Naive Bayes classifier for
numerical data.  The class offers standard classification functionality.  Naive
Bayes is useful for multi-class classification (i.e. classes are `0`, `1`, `2`,
etc.), and due to its simplicity scales well to large-data scenarios.

#### Simple usage example:

```c++
// Train a Naive Bayes classifier on random data and predict labels:

// All data and labels are uniform random; 5 dimensional data, 4 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(5, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 3));
arma::mat testDataset(5, 500, arma::fill::randu); // 500 test points.

mlpack::NaiveBayesClassifier nbc;       // Step 1: create model.
nbc.Train(dataset, labels, 4);          // Step 2: train model.
arma::Row<size_t> predictions;
nbc.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `NaiveBayesClassifier` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-different-element-types) for
   using different element types for a model.

#### See also:

 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [`GaussianDistribution`](../core/distributions.md#gaussiandistribution)
 * [Naive Bayes classifier on Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

### Constructors

 * `nbc = NaiveBayesClassifier()`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Classify()`](#classification).

---

 * `nbc = NaiveBayesClassifier(dimensionality, numClasses, epsilon=1e-10)`
   - Initialize model to the given dimensionality and number of classes without
     training.
   - This is meant to be used with the incremental version of `Train()` that
     takes only a single point.

---

 * `nbc = NaiveBayesClassifier(data, labels, numClasses, incremental=true, epsilon=1e-10)`
   - Train model, optionally specifying whether to do incremental training.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `incremental` | `bool` | If `true`, then the model will not be reset before training, and will use a robust incremental algorithm for variance computation. | `true` |
| `epsilon` | `double` | Initial small value for sample variances, to prevent underflow (via `log(0)`). | 1e-10 |

As an alternative to passing the `epsilon` parameter, it can be set with the
standalone `Epsilon()` method: `nbc.Epsilon() = eps;` will set the value of
`epsilon` to `eps` for the next time non-incremental `Train()` or `Reset()` is
called.

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

 * `nbc.Train(data, labels, numClasses, incremental=true, epsilon=1e-10)`
   - Train model on the given data, optionally specifying whether to do
     incremental training.
   - Arguments described in [Constructor Parameters](#constructor-parameters)
     table above.

---

 * `nbc.Train(point, label)`
   - Incrementally train on a single data point with the given label.
   - Ensure that the model has the right size and number of classes by using the
     appropriate constructor form to set `dimensionality`, or by calling
     `Reset()` (see [other functionality](#other-functionality)).

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `point` | [`arma::vec`](../matrices.md) | [Column-major](../matrices.md) training point (i.e. one column). | _(N/A)_ |
| `label` | `size_t` | Training label, in range `0` to `numClasses`. | _(N/A)_ |

***Note***: when performing incremental training, if `data` has a different
dimensionality than the model, or if `numClasses` is different, the model will
be reset.  For single-point `Train()`, if `point` has different dimensionality,
an exception will be thrown.

### Classification

Once a `NaiveBayesClassifier` model is trained, the `Classify()` member function
can be used to make class predictions for new data.

 * `size_t predictedClass = nbc.Classify(point)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class (`0` through
     `numClasses - 1`, inclusive).

---

 * `nbc.Classify(point, prediction, probabilitiesVec)`
   - ***(Single-point)***
   - Classify a single point and compute class probabilities.
   - The predicted class is stored in `prediction`.
   - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `nbc.Classify(data, predictions)`
   - ***(Multi-point)***
   - Classify a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `nbc.Classify(data, predictions, probabilities)`
   - ***(Multi-point)***
   - Classify a set of points and compute class probabilities.
   - The prediction for data point `i` can be accessed with `predictions[i]`.
   - The probability of class `j` for data point `i` can be accessed with
     `probabilities(j, i)`.

---

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
| _single-point_ | `probabilitiesVec` | [`arma::vec&`](../matrices.md) | `arma::vec&` to store class probabilities into; will have length 2. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into; will be set to length `data.n_cols`. |
| _multi-point_ | `probabilities` | [`arma::mat&`](../matrices.md) | Matrix to store class probabilities into (number of rows will be equal to 2; number of columns will be equal to `data.n_cols`). |

### Other Functionality

 * A `NaiveBayesClassifier` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `nbc.Probabilities()` will return a column vector of length `numClasses`
   representing the prior probability of each class.

 * `nbc.Means()` will return a matrix with rows equal to the dimensionality of
   the model and `numClasses` columns.  Column `i` represents the sample mean of
   class `i`.

 * `nbc.Variances()` will return a matrix with rows equal to the dimensionality
   of the model and `numClasses` columns.  The element at row `i` and column `j`
   represents the sample variance in dimension `i` of class `j`.

 * `nbc.Reset()` will reset the model to zeros; this is useful before
   incremental training.  The form
   `nbc.Reset(dimensionality, numClasses, epsilon=1e-10)` can
   also be used to set the dimensionality and number of classes in the reset
   model.

 * `nbc.TrainingPoints()` will return the number of points that the model has
   been trained on.  When `nbc.Reset()` is called, this is reset to 0.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `NaiveBayesClassifier` class.

---

Train a Naive Bayes classifier incrementally, one point at a time, then compute
accuracy on a test set and save the model to disk.

```c++
// See https://datasets.mlpack.org/mnist.train.csv.
arma::mat dataset;
mlpack::data::Load("mnist.train.csv", dataset, true);
// See https://datasets.mlpack.org/mnist.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("mnist.train.labels.csv", labels, true);

mlpack::NaiveBayesClassifier nbc(dataset.n_rows /* dimensionality */,
                                 10 /* numClasses */);

// Iterate over all points in the dataset and call Train() on each point.
for (size_t i = 0; i < dataset.n_cols; ++i)
  nbc.Train(dataset.col(i), labels[i]);

// Now compute the accuracy of the fully trained model on a test set.

// See https://datasets.mlpack.org/mnist.test.csv.
arma::mat testDataset;
mlpack::data::Load("mnist.test.csv", testDataset, true);
// See https://datasets.mlpack.org/mnist.test.labels.csv.
arma::Row<size_t> testLabels;
mlpack::data::Load("mnist.test.labels.csv", testLabels, true);

arma::Row<size_t> predictions;
nbc.Classify(dataset, predictions);
const double trainAccuracy = 100.0 *
    ((double) arma::accu(predictions == labels)) / labels.n_elem;
std::cout << "Accuracy of model on training data: " << trainAccuracy << "\%."
    << std::endl;

nbc.Classify(testDataset, predictions);

const double testAccuracy = 100.0 *
    ((double) arma::accu(predictions == testLabels)) / testLabels.n_elem;
std::cout << "Accuracy of model on test data:     " << testAccuracy << "\%."
    << std::endl;

// Save the model to disk with the name "nbc".
mlpack::data::Save("nbc_model.bin", "nbc", nbc, true);
```

---

Load a saved Naive Bayes classifier and print some information about it.

```c++
mlpack::NaiveBayesClassifier nbc;

// Load the model named "nbc" from "nbc_model.bin".
mlpack::data::Load("nbc_model.bin", "nbc", nbc, true);

// Print information about the model.
std::cout << "The dimensionality of the model in nbc_model.bin is "
    << nbc.Means().n_rows << "." << std::endl;
std::cout << "The number of classes in the model is "
    << nbc.Probabilities().n_elem << "." << std::endl;
std::cout << "The model was trained on " << nbc.TrainingPoints() << " points."
    << std::endl;
std::cout << "The prior probabilities of each class are: "
    << nbc.Probabilities().t();

// Compute the class probabilities of a random point.
// For our random point, we'll use one of the means plus some noise.
arma::vec randomPoint = nbc.Means().col(2) +
    10.0 * arma::randu<arma::vec>(nbc.Means().n_rows);

size_t prediction;
arma::vec probabilities;
nbc.Classify(randomPoint, prediction, probabilities);

std::cout << "Random point class prediction: " << prediction << "."
    << std::endl;
std::cout << "Random point class probabilities: " << probabilities.t();
```

---

See also the following fully-working examples:

 - [Microchip QA Classification using `NaiveBayesClassifier`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/naive_bayes/microchip_quality_control/microchip-quality-control-cpp.ipynb)

### Advanced Functionality: Different Element Types

The `NaiveBayesClassifier` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```
NaiveBayesClassifier<ModelMatType>
```

`ModelMatType` specifies the type of matrix used for training data and internal
representation of model parameters.

 * Any matrix type that implements the Armadillo API can be used.

 * `Train()` and `Classify()` functions themselves are templatized and can allow
   any matrix type that has the same element type.  So, for instance, a
   `NaiveBayesClassifier<arma::mat>` can accept an `arma::sp_mat` for training.

The example below trains a Naive Bayes model on sparse 32-bit floating point
data, but uses dense 32-bit floating point matrices to store the model itself.

```c++
// Create random, sparse 100-dimensional data, with 3 classes.
arma::sp_fmat dataset;
dataset.sprandu(100, 5000, 0.3);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(5000, arma::distr_param(0, 2));

mlpack::NaiveBayesClassifier<arma::fmat> nbc(dataset, labels, 3);

// Now classify a test point.
arma::sp_fvec point;
point.sprandu(100, 1, 0.3);

size_t prediction;
arma::fvec probabilitiesVec;
nbc.Classify(point, prediction, probabilitiesVec);

std::cout << "Prediction for random test point: " << prediction << "."
    << std::endl;
std::cout << "Class probabilities for random test point: "
    << probabilitiesVec.t();
```

***Note:*** dense objects should be used for `ModelMatType`, since in general
the mean and sample variance of sparse data is dense.
