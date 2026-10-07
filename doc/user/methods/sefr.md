## `SEFR`

The `SEFR` class implements SEFR (Scalable, Efficient, and Fast classifieR), a
linear classifier whose training time is linear in the size of the data and
that has no hyperparameters.  For each class, SEFR compares the average of the
points in that class with the average of all other points; each feature's
weight is the normalized difference of those two averages, and the bias places
the decision boundary between the average scores of the two groups.
Multi-class problems are handled one-vs-all.

SEFR is useful for classifying points with _discrete labels_ (i.e., `0`, `1`,
`2`) when training must be very cheap, for instance on low-resource or embedded
hardware, or when the model must be updated incrementally as new points arrive.
It accepts any numeric data, and works well on sparse non-negative data such
as counts or TF-IDF features.  Because it supports instance weights, it can also
be used as a weak learner for [`AdaBoost`](adaboost.md).

#### Simple usage example:

```c++
// Train a SEFR model on random numeric data and predict labels on test data:

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::SEFR s;                       // Step 1: create model.
s.Train(dataset, labels, 5);          // Step 2: train model.
arma::Row<size_t> predictions;
s.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 1) << " test points classified as class "
    << "1." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `SEFR` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.
 * [Advanced template examples](#advanced-functionality-examples) of use with
   custom template parameters.

#### See also:

 * [`NaiveBayesClassifier`](naive_bayes_classifier.md), another simple
   classifier that can be trained incrementally
 * [`Perceptron`](perceptron.md)
 * [`LinearSVM`](linear_svm.md)
 * [`AdaBoost`](adaboost.md)
 * [mlpack classifiers](../modeling.md#classification)
 * [SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices (pdf)](https://arxiv.org/pdf/2006.04620)

### Constructors

Construct a `SEFR` object using one of the constructors below.  Defaults and
types are detailed in the [Constructor Parameters](#constructor-parameters)
section below.

#### Forms:

 * `s = SEFR()`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Classify()`](#classification).

---

 * `s = SEFR(numClasses, dimensionality)`
   - Initialize the model with all-zero weights and biases.
   - The model can then be trained one point at a time with the
     [single-point `Train()`](#training) overload.

---

 * `s = SEFR(data, labels, numClasses)`
 * `s = SEFR(data, labels, numClasses, weights)`
   - Train the model (optionally with instance weights).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, between [`0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`](../matrices.md) | Weights for each training point.  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset.  Must be at least 2. | _(N/A)_ |
| `dimensionality` | `size_t` | Dimensionality of data (only used if an initialized but untrained model is desired). | _(N/A)_ |

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `s.Train(data, labels, numClasses)`
   - Train the model on unweighted data.

---

 * `s.Train(data, labels, numClasses, weights)`
   - Train the model on data with instance weights.

---

 * `s.Train(point, label)`
   - Incrementally update the model with a single point.
   - The model must already have the right number of classes and
     dimensionality, either from a previous call to `Train()` or from the
     `SEFR(numClasses, dimensionality)` constructor.
   - The result is identical to training on all points seen so far at once.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters); `point` is an
[`arma::vec`](../matrices.md) and `label` is a `size_t`.

***Notes***:

 * Training on a dataset (the first two forms) replaces any existing model.
   Training on a single point (the third form) updates the existing model.

 * Training only computes per-class sums and counts of the data, so it takes
   a single pass over the data; the cost of the single-point form is
   proportional to `dimensionality * numClasses`.

 * SEFR's weight formula assumes non-negative features.  Data with negative
   values is handled automatically: before computing the weights, each feature
   is shifted by its minimum over the training data when that minimum is
   negative.  The shift is computed in the same single pass, is folded into the
   biases (so prediction is unaffected), and keeps sparse data sparse.
   Features that are already non-negative are not shifted, so results on
   non-negative data are the same as in the SEFR paper.

 * SEFR uses the features as given, so their scales still matter; scaling the
   data, for instance with `mlpack::MinMaxScaler` as in the
   [iris example](#simple-examples) below, can improve accuracy.

### Classification

Once a `SEFR` model is trained, the `Classify()` member function can be used to
make class predictions for new data.

 * `size_t predictedClass = s.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `s.Classify(point, prediction, scores)`
    - ***(Single-point)***
    - Classify a single point and compute class scores.
    - The predicted class is stored in `prediction`.
    - The score for class `j` can be accessed with `scores[j]`.

---

 * `s.Classify(data, predictions)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `s.Classify(data, predictions, scores)`
    - ***(Multi-point)***
    - Classify a set of points and compute class scores for each point.
    - The prediction for data point `i` can be accessed with `predictions[i]`.
    - The score for class `j` for data point `i` can be accessed with
      `scores(j, i)`.

---

***Note***: scores are not probabilities and are not normalized to `[0, 1]`.
The score of class `j` for a point `x` is `s.Weights().col(j)' * x +
s.Biases()[j]`: it is positive on class `j`'s side of that class's
one-vs-rest hyperplane, and it equals the signed distance from that hyperplane
multiplied by the norm of `s.Weights().col(j)`.  Its magnitude therefore
depends on the scale of the features, and scores of different classes are not
calibrated against each other.  The predicted class is the one with the highest
score.  For two classes, the scores of the two classes are negatives of each
other.  This is the same kind of score that [`LinearSVM`](linear_svm.md)
returns; classifiers such as
[`LogisticRegression`](logistic_regression.md) and
[`NaiveBayesClassifier`](naive_bayes_classifier.md) return class probabilities
instead.

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
| _single-point_ | `scores` | [`arma::vec&`](../matrices.md) | `arma::vec&` to store class scores into; will be set to length `numClasses`. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into.  Will be set to length `data.n_cols`. |
| _multi-point_ | `scores` | [`arma::mat&`](../matrices.md) | Matrix to store class scores into (number of rows will be equal to number of classes, number of columns will be equal to number of points). |

### Other Functionality

 * A `SEFR` model can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).  A
   loaded model can still be updated incrementally.

 * `s.NumClasses()` will return a `size_t` indicating the number of classes the
   model was trained on.

 * `s.Dimensionality()` will return a `size_t` indicating the dimensionality of
   the model.

 * `s.Weights()` will return an `arma::mat` with the weights of the model (each
   column corresponds to the weights for one class label).

 * `s.Biases()` will return an `arma::vec` with the biases of the model (each
   element corresponds to the bias for a class).  The score of class `j` for a
   point `x` is `s.Weights().col(j)' * x + s.Biases()[j]`.

 * `s.ClassSums()` will return an `arma::mat` with the (weighted) sum of the
   training points of each class, and `s.ClassCounts()` will return an
   `arma::vec` with the (weighted) number of training points of each class.

 * `s.Offsets()` will return an `arma::vec` with the shift applied to each
   feature before computing the weights: the minimum of that feature over the
   training data if it is negative, and `0` otherwise.

 * `s.Reset()` will set all weights, biases, sums, counts, and offsets to zero,
   keeping the number of classes and dimensionality.

For complete functionality, the source code in
`src/mlpack/methods/sefr/sefr.hpp` can be consulted.  Each method is fully
documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`SEFR`.

---

Train a model on the iris dataset, print its accuracy on a held-out test set,
and save it to disk.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::Load("iris.csv", dataset, mlpack::Fatal);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::Load("iris.labels.csv", labels, mlpack::Fatal);

// Scale each feature to [0, 1].
mlpack::MinMaxScaler scaler;
scaler.Fit(dataset);
scaler.Transform(dataset, dataset);

// Split into a training set and a test set.
arma::mat trainData, testData;
arma::Row<size_t> trainLabels, testLabels;
mlpack::Split(dataset, labels, trainData, testData, trainLabels, testLabels,
    0.3);

mlpack::SEFR s(trainData, trainLabels, 3);

arma::Row<size_t> predictions;
s.Classify(testData, predictions);
std::cout << "Test set accuracy: "
    << (100.0 * double(arma::accu(testLabels == predictions)) /
        testLabels.n_elem) << "\%." << std::endl;

// Save the model to disk for later use.
mlpack::Save("sefr.bin", s);
```

---

Train a model one point at a time, as points arrive in a stream, and check its
accuracy periodically.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::Load("iris.csv", dataset, mlpack::Fatal);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::Load("iris.labels.csv", labels, mlpack::Fatal);

// Shuffle the points so that classes are interleaved.
mlpack::ShuffleData(dataset, labels, dataset, labels);

// Create an untrained model with 3 classes and the right dimensionality.
mlpack::SEFR s(3, dataset.n_rows);

arma::Row<size_t> predictions;
for (size_t i = 0; i < dataset.n_cols; ++i)
{
  s.Train(dataset.col(i), labels[i]);

  if ((i + 1) % 50 == 0)
  {
    s.Classify(dataset, predictions);
    std::cout << "Accuracy after " << (i + 1) << " points: "
        << (100.0 * double(arma::accu(labels == predictions)) / labels.n_elem)
        << "\%." << std::endl;
  }
}
```

---

Load a saved model from disk and print information about it.

```c++
mlpack::SEFR s;
// This call assumes a model has already been saved to `sefr.bin` with
// `Save()`.
mlpack::Load("sefr.bin", s, mlpack::Fatal);

if (s.NumClasses() > 0)
{
  std::cout << "The model in `sefr.bin` was trained on " << s.NumClasses()
      << " classes." << std::endl;
  std::cout << "The dimensionality of the model is " << s.Dimensionality()
      << "." << std::endl;
  for (size_t i = 0; i < s.NumClasses(); ++i)
  {
    std::cout << "  - Class " << i << ": " << s.ClassCounts()[i]
        << " training points, bias " << s.Biases()[i] << "." << std::endl;
  }
}
else
{
  std::cout << "The model in `sefr.bin` has not been trained." << std::endl;
}
```

---

Use `SEFR` as the weak learner for [`AdaBoost`](adaboost.md).

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::Load("iris.csv", dataset, mlpack::Fatal);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::Load("iris.labels.csv", labels, mlpack::Fatal);

// Train AdaBoost with up to 25 SEFR weak learners.
mlpack::AdaBoost<mlpack::SEFR<>> a(dataset, labels, 3, 25);

arma::Row<size_t> predictions;
a.Classify(dataset, predictions);
std::cout << "AdaBoost used " << a.WeakLearners() << " weak learners; training "
    << "set accuracy: "
    << (100.0 * double(arma::accu(labels == predictions)) / labels.n_elem)
    << "\%." << std::endl;
```

---

### Advanced Functionality: Template Parameters

The `SEFR` class has one template parameter, which can be used for custom
behavior.  The full signature of the class is as follows:

```
SEFR<ModelMatType>
```

 * `ModelMatType`: specifies the type of matrix used to represent the model and
   the data.

---

#### `ModelMatType`

 * Specifies the matrix type used for the weights, biases, and per-class
   statistics of the model.
 * By default, `ModelMatType` is `arma::mat` (dense 64-bit precision matrix).
 * Any dense or sparse matrix type implementing the Armadillo API will work; so,
   for instance, `arma::fmat` or `arma::sp_fmat` can be used.  The model itself
   is always stored densely.
 * The element type of the data given to `Train()` and `Classify()` must match
   the element type of `ModelMatType` (e.g. `arma::fmat` data for an
   `SEFR<arma::fmat>` model).

### Advanced Functionality Examples

Train a `SEFR` model on sparse 32-bit floating point data, such as word counts.

```c++
// 1000 sparse random points in 5000 dimensions, with 1% nonzero elements.
arma::sp_fmat dataset;
dataset.sprandu(5000, 1000, 0.01);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor.
mlpack::SEFR<arma::sp_fmat> s(dataset, labels, 5);

// Create test data (500 points).
arma::sp_fmat testDataset;
testDataset.sprandu(5000, 500, 0.01);
arma::Row<size_t> predictions;
s.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 1) << " test points classified as class "
    << "1." << std::endl;
```

---
