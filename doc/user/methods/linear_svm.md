## `LinearSVM`

The `LinearSVM` class implements an L2-regularized support vector machine for
numerical data that can train using any ensmallen optimizer.  The class offers
standard classification functionality.  Linear SVM is useful for multi-class
classification (i.e. classes are `0`, `1`, `2`, etc.).

#### Simple usage example:

```c++
// Train a linear SVM classifier on random data and predict labels:

// All data and labels are uniform random; 5 dimensional data, 4 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(5, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 3));
arma::mat testDataset(5, 500, arma::fill::randu); // 500 test points.

mlpack::LinearSVM svm;                  // Step 1: create model.
svm.Train(dataset, labels, 4);          // Step 2: train model.
arma::Row<size_t> predictions;
svm.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 1) << " test points classified as class "
    << "1." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `LinearSVM` objects.
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

 * `svm = LinearSVM()`
   - Initialize the parameters of the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Classify()`](#classification).

---

 * `svm = LinearSVM(dimensionality, numClasses, lambda=0.0001, delta=1.0, fitIntercept=false)`
   - Initialize the model without training, to default weights.
   - [`Classify()`](#classification) can immediately be called and
     `Parameters()` returns valid weights, but the model is otherwise untrained.
   - The model should be trained with [`Train()`](#training) before calling
     [`Classify()`](#classification).

---

 * `svm = LinearSVM(data, labels, numClasses, lambda=0.0001, delta=1.0, fitIntercept=false, [callbacks...])`
   - Train model, optionally specifying ensmallen callbacks for use during
     optimization.

---

 * `svm = LinearSVM(data, labels, numClasses, optimizer, lambda=0.0001, delta=1.0, fitIntercept=false, [callbacks...])`
   - Train model with a custom ensmallen optimizer, optionally specifying
     callbacks for use during optimization.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `dimensionality` | `size_t` | Dimension of input data (if data is not specified).  Should be equal to `data.n_rows`. | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `optimizer` | [any ensmallen optimizer](https://www.ensmallen.org) | Instantiated ensmallen optimizer for [differentiable functions](https://www.ensmallen.org/docs.html#differentiable-functions) or [differentiable separable functions](https://www.ensmallen.org/docs.html#differentiable-separable-functions). | `ens::L_BFGS()` |
| `lambda` | `double` | L2 regularization penalty parameter.  Must be nonnegative. | `0.0` |
| `delta` | `double` | Margin of difference between correct class and other classes. | `1.0` |
| `fitIntercept` | `bool` | If `true`, then an intercept term is fitted to the model. | `false` |
| `callbacks...` | [any set of ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation) | Optional callbacks for the ensmallen optimizer, such as e.g. `ens::ProgressBar()`, `ens::Report()`, or others. | _(N/A)_ |

As an alternative to passing `lambda`, `delta`, or `fitIntercept`, these can be
set with a standalone method.  The following functions can be used before
calling `Train()`:

 * `svm.Lambda() = lambda;` will set the L2 regularization penalty parameter to
   `lambda`.
 * `svm.Delta() = delta;` will set the margin of difference to `delta`.
 * `svm.FitIntercept() = fitIntercept;` will set whether the model fits an
   intercept to `fitIntercept`.

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

 * `svm.Train(data, labels, numClasses,            [callbacks...])`
 * `svm.Train(data, labels, numClasses, optimizer, [callbacks...])`
   - Train model without changing any hyperparameters, optionally using a custom
     ensmallen optimizer and specifying callbacks for use during optimization.

---

 * `svm.Train(data, labels, numClasses,            lambda=0.0001, delta=1.0, fitIntercept=false, [callbacks...])`
 * `svm.Train(data, labels, numClasses, optimizer, lambda=0.0001, delta=1.0, fitIntercept=false, [callbacks...])`
   - Train model on the given data, specifying hyperparameters and optionally
     also a custom ensmallen optimizer and callbacks for use during
     optimization.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Note:*** Training is not incremental.  Successive calls to `Train()` will
train entirely new models.

### Classification

Once a `LinearSVM` model is trained, the `Classify()` member function
can be used to make class predictions for new data.

 * `size_t predictedClass = svm.Classify(point)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class (`0` through
     `numClasses - 1`, inclusive).

---

 * `svm.Classify(point, prediction, probabilitiesVec)`
   - ***(Single-point)***
   - Classify a single point and compute class probabilities.
   - The predicted class is stored in `prediction`.
   - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `svm.Classify(data, predictions)`
   - ***(Multi-point)***
   - Classify a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `svm.Classify(data, predictions, probabilities)`
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
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md) points for classification. |
| _multi-point_ | `predictions` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into; will be set to length `data.n_cols`. |
| _multi-point_ | `probabilities` | [`arma::mat&`](../matrices.md) | Matrix to store class probabilities into (number of rows will be equal to 2; number of columns will be equal to `data.n_cols`). |

### Other Functionality

 * A `LinearSVM` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `svm.Parameters()` will return the parameters of the model as an `arma::mat`
   with either `data.n_rows` rows (if `FitIntercept()` is `false`) or
   `data.n_rows + 1` rows (if `FitIntercept()` is `true`), and `numClasses`
   columns.  The weight for dimension `i` for class `j` can be accessed with
   `svm.Parameters()(i, j)`.  If `FitIntercept()` is `true`, the last row of
   `svm.Parameters()` represents the bias parameters for each class.

 * `svm.FeatureSize()` will return the number of features in the model.  This is
   equivalent to `data.n_rows` when the model was trained.  The output is only
   valid if the model has been trained.

 * `svm.ComputeAccuracy(data, labels)` will return the accuracy of the model on
   the given `data` with the given `labels`.  The returned accuracy is between 0
   and 100.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `LinearSVM` class.

---

Train a linear SVM using a custom SGD-like optimizer with callbacks.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, true);
// See https://datasets.mlpack.org/satellite.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.train.labels.csv", labels, true);

mlpack::LinearSVM svm;
svm.Lambda() = 0.1;

// Create AMSGrad optimizer with custom step size and batch size.
ens::AMSGrad optimizer(0.01 /* step size */, 16 /* batch size */);
optimizer.MaxIterations() = 100 * dataset.n_cols; // Allow 100 epochs.

// Print a progress bar and an optimization report when training is finished.
svm.Train(dataset, labels, 2, optimizer, ens::ProgressBar(), ens::Report());

// Now predict on test labels and compute accuracy.

// See https://datasets.mlpack.org/satellite.test.csv.
arma::mat testDataset;
mlpack::data::Load("satellite.test.csv", testDataset, true);
// See https://datasets.mlpack.org/satellite.test.labels.csv.
arma::Row<size_t> testLabels;
mlpack::data::Load("satellite.test.labels.csv", testLabels, true);

std::cout << std::endl;
std::cout << "Accuracy on training set: "
    << svm.ComputeAccuracy(dataset, labels) << "\%." << std::endl;
std::cout << "Accuracy on test set:     "
    << svm.ComputeAccuracy(testDataset, testLabels) << "\%." << std::endl;
```

---

Train a linear SVM with SGD and save the model every epoch using a [custom
ensmallen callback](https://www.ensmallen.org/docs.html#custom-callbacks):

```c++
// This callback saves the model into "model-<epoch>.bin" after every epoch.
class ModelCheckpoint
{
 public:
  ModelCheckpoint(mlpack::LinearSVM<>& model) : model(model) { }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double /* objective */)
  {
    const std::string filename = "model-" + std::to_string(epoch) + ".bin";
    mlpack::data::Save(filename, "svm", model, true);
    return false; // Do not terminate the optimization.
  }

 private:
  mlpack::LinearSVM<>& model;
};
```

With that callback available, the code to train the model is below:

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, true);
// See https://datasets.mlpack.org/satellite.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.train.labels.csv", labels, true);

mlpack::LinearSVM svm;

// Create AdaDelta optimizer with a small step size and batch size of 1.
ens::AdaDelta adaDelta(0.001, 1);
adaDelta.MaxIterations() = 100 * dataset.n_cols; // 100 epochs maximum.

// Use the custom callback and an L2 penalty parameter of 0.01, with default
// delta and fitting an intercept.
svm.Train(dataset, labels, 2, adaDelta, 0.01, 1.0, true, ModelCheckpoint(svm),
    ens::ProgressBar());

// Now files like model-1.bin, model-2.bin, etc. should be saved on disk.
```

---

Load a linear SVM from disk and print some information about it.

```c++
mlpack::LinearSVM svm;
// This assumes that a model called "svm" has been saved to the file
// "model-1.bin" (as in the previous example).
mlpack::data::Load("model-1.bin", "svm", svm, true);

// Print the dimensionality of the model and some other statistics.
std::cout << "The dimensionality of the model in model-1.bin is "
    << svm.FeatureSize() << "." << std::endl;
if (svm.FitIntercept())
{
  std::cout << "Intercept values for each class: " << std::endl;
  for (size_t i = 0; i < svm.Parameters().n_cols; ++i)
  {
    std::cout << "  - Class " << i << ": "
        << svm.Parameters()(svm.Parameters().n_rows - 1, i) << "." << std::endl;
  }
}
else
{
  std::cout << "The model does not have an intercept fitted." << std::endl;
}

std::cout << "The L2 regularization penalty parameter is: " << svm.Lambda()
    << "." << std::endl;

std::cout << "Weights for the first dimension are: "
    << svm.Parameters().row(0);
```

---

### Advanced Functionality: Different Element Types

The `LinearSVM` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```
LinearSVM<ModelMatType>
```

`ModelMatType` specifies the type of matrix used for training data and internal
representation of model parameters.

 * Any matrix type that implements the Armadillo API can be used.

 * `Train()` and `Classify()` functions themselves are templatized and can allow
   any matrix type that has the same element type.  So, for instance, a
   `LinearSVM<arma::mat>` can accept an `arma::sp_mat` for training.

The example below trains a linear SVM on sparse 32-bit floating point
data, but uses dense 32-bit floating point matrices to store the model itself.

```c++
// Create random, sparse 100-dimensional data, with 3 classes.
arma::sp_fmat dataset;
dataset.sprandu(100, 5000, 0.3);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(5000, arma::distr_param(0, 2));

mlpack::LinearSVM<arma::fmat> svm(dataset, labels, 3);

// Now classify a test point.
arma::sp_fvec point;
point.sprandu(100, 1, 0.3);

size_t prediction;
arma::fvec probabilitiesVec;
svm.Classify(point, prediction, probabilitiesVec);

std::cout << "Prediction for random test point: " << prediction << "."
    << std::endl;
std::cout << "Class probabilities for random test point: "
    << probabilitiesVec.t();
```

***Note:*** dense objects should be used for `ModelMatType`, since in general
L2-regularized models are fully dense.
