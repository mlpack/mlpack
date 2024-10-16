## `LogisticRegression`

The `LogisticRegression` class implements a simple L2-regularized two-class
logistic regression classifier for numerical data, by default using L-BFGS to
learn the model.  The class offers easy configurability, and arbitrary
optimizers can be used to learn the model.

Logistic regression is useful for two-class classification (i.e. classes are `0`
or `1`).  For multi-class logistic regression, see
[`SoftmaxRegression`](softmax_regression.md).

#### Simple usage example:

```c++
// Train a logistic regression model on random data and predict labels:

// All data and labels are uniform random; 5 dimensional data, 2 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(5, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 1));
arma::mat testDataset(5, 500, arma::fill::randu); // 500 test points.

mlpack::LogisticRegression lr;         // Step 1: create model.
lr.Train(dataset, labels);             // Step 2: train model.
arma::Row<size_t> predictions;
lr.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 0) << " test points classified as class "
    << "0." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `LogisticRegression` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-different-element-types) for
   using different element types for a model.

#### See also:

 * [`SoftmaxRegression`](softmax_regression.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [Logistic regression on Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)

### Constructors

 * `lr = LogisticRegression()`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Classify()`](#classification).

---

 * `lr = LogisticRegression(data, labels,               lambda=0.0, [callbacks...])`
 * `lr = LogisticRegression(data, labels, initialPoint, lambda=0.0, [callbacks...])`
   - Train model, optionally specifying an initial set of weights for the
     optimization and callbacks.

---

 * `lr = LogisticRegression(data, labels, optimizer,               lambda=0.0, [callbacks...])`
 * `lr = LogisticRegression(data, labels, optimizer, initialPoint, lambda=0.0, [callbacks...])`
   - Train model with a custom ensmallen optimizer, optionally specifying an
     initial set of weights to start the optimization from and callbacks for the
     optimizer.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, either [`0` or `1`](../core/normalizing_labels.md).  Should have length `data.n_cols`.  | _(N/A)_ |
| `initialPoint` | `arma::rowvec` | Initial model weights to start optimization from.  Should have length `data.n_rows + 1`.  The first element is the bias.  If not specified, a zero vector will be used. | zero vector |
| `optimizer` | [any ensmallen optimizer](https://www.ensmallen.org) | Instantiated ensmallen optimizer for [differentiable functions](https://www.ensmallen.org/docs.html#differentiable-functions) or [differentiable separable functions](https://www.ensmallen.org/docs.html#differentiable-separable-functions). | `ens::L_BFGS()` |
| `lambda` | `double` | L2 regularization penalty parameter.  Must be nonnegative. | `0.0` |
| `callbacks...` | [any set of ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation) | Optional callbacks for the ensmallen optimizer, such as e.g. `ens::ProgressBar()`, `ens::Report()`, or others. | _(N/A)_ |

As an alternative to passing `lambda` or `initialPoint`, these can be set with a
standalone method.  The following functions can be used before calling
`Train()`:

 * `lr.Lambda() = l;` will set the value of the L2 regularization penalty
   parameter to `l`.
 * `lr.Parameters() = initialPoint;` will set the initial point for the training
   optimization to `initialPoint`.

***Note***: Setting `lambda` too small may cause the model to overfit; however,
setting it too large may cause the model to underfit.  [Automatic hyperparameter
tuning](../hpt.md) can be used to find a good value of `lambda` instead of a
manual setting.

<!-- TODO: fix link for hyperparameter tuner -->

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `lr.Train(data, labels)`
 * `lr.Train(data, labels, lambda=0.0, [callbacks...])`
   - Train model, optionally specifying callbacks for the default L-BFGS
     optimizer.

---

 * `lr.Train(data, labels, optimizer)`
 * `lr.Train(data, labels, optimizer, lambda=0.0, [callbacks...])`
   - Train model with a custom ensmallen optimizer, optionally specifying
     callbacks.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes***:

 * Training is incremental.  Successive calls to `Train()` will not reinitialize
   the model, unless the given data has different dimensionality.  To
   reinitialize the model, call `Reset()` (see
   [Other Functionality](#other-functionality)).

 * To set the initial point of the optimization, call `Parameters()`; see
   [Other Functionality](#other-functionality).

 * `Train()` returns a `double` with the final logistic regression loss value
   (including L2 penalty term) of the trained model.

### Classification

Once a `LogisticRegression` model is trained, the `Classify()` member function
can be used to make class predictions for new data.

 * `size_t predictedClass = lr.Classify(point, decisionBoundary=0.5)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class (`0` or `1`).

---

 * `lr.Classify(point, prediction, probabilitiesVec, decisionBoundary=0.5)`
   - ***(Single-point)***
   - Classify a single point and compute class probabilities.
   - The predicted class is stored in `prediction`.
   - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `lr.Classify(data, predictions, decisionBoundary=0.5)`
   - ***(Multi-point)***
   - Classify a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `lr.Classify(data, predictions, probabilities, decisionBoundary=0.5)`
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
||||
| _all_ | `decisionBoundary` | `double` | If the logistic function value for a point is greater than `decisionBoundary`, it is classified as class `1`.  Defaults to `0.5`. |

### Other Functionality

 * A `LogisticRegression` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `lr.Parameters()` will return an `arma::rowvec` filled with the weights of
   the model.  This vector has length equal to the dimensionality plus one, and
   the first element is the bias.

 * `lr.Lambda()` will return the L2 regularization penalty parameter.

 * `lr.ComputeAccuracy(data, labels, decisionBoundary=0.5)` will return the
   accuracy of the model on the given `data` with the given `labels`.  The
   returned accuracy is between 0 and 100.

 * `lr.ComputeError(data, labels)` will return the loss of the logistic
   regression objective function on the given `data` with the given `labels`.

 * `lr.Reset()` will reset the weights of the model to zeros.

For complete functionality, the [source
code](/src/mlpack/methods/logistic_regression/logistic_regression.hpp) can be
consulted.  Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `LogisticRegression` class.

---

Train a logistic regression model using a custom SGD-like optimizer with
callbacks.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, true);
// See https://datasets.mlpack.org/satellite.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.train.labels.csv", labels, true);

mlpack::LogisticRegression lr;
lr.Lambda() = 0.1;

// Create AMSGrad optimizer with custom step size and batch size.
ens::AMSGrad optimizer(0.01 /* step size */, 16 /* batch size */);
optimizer.MaxIterations() = 100 * dataset.n_cols; // Allow 100 epochs.

// Print a progress bar and an optimization report when training is finished.
lr.Train(dataset, labels, optimizer, ens::ProgressBar(), ens::Report());

// Now predict on test labels and compute accuracy.

// See https://datasets.mlpack.org/satellite.test.csv.
arma::mat testDataset;
mlpack::data::Load("satellite.test.csv", testDataset, true);
// See https://datasets.mlpack.org/satellite.test.labels.csv.
arma::Row<size_t> testLabels;
mlpack::data::Load("satellite.test.labels.csv", testLabels, true);

std::cout << std::endl;
std::cout << "Accuracy on training set: "
    << lr.ComputeAccuracy(dataset, labels) << "\%." << std::endl;
std::cout << "Accuracy on test set:     "
    << lr.ComputeAccuracy(testDataset, testLabels) << "\%." << std::endl;
std::cout << "Objective on training set: "
    << lr.ComputeError(dataset, labels) << "." << std::endl;
std::cout << "Objective on test set:     "
    << lr.ComputeError(testDataset, testLabels) << "." << std::endl;
```

---

Train a logistic regression model with SGD and save the model every epoch using
a [custom ensmallen
callback](https://www.ensmallen.org/docs.html#custom-callbacks):

```c++
// This callback saves the model into "model-<epoch>.bin" after every epoch.
class ModelCheckpoint
{
 public:
  ModelCheckpoint(mlpack::LogisticRegression<>& model) : model(model) { }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double /* objective */)
  {
    const std::string filename = "model-" + std::to_string(epoch) + ".bin";
    mlpack::data::Save(filename, "lr_model", model, true);
    return false; // Do not terminate the optimization.
  }

 private:
  mlpack::LogisticRegression<>& model;
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

mlpack::LogisticRegression lr;

// Create AdaDelta optimizer with a small step size and batch size of 1.
ens::AdaDelta adaDelta(0.001, 1);
adaDelta.MaxIterations() = 100 * dataset.n_cols; // 100 epochs maximum.

// Use the custom callback and an L2 penalty parameter of 0.01.
lr.Train(dataset, labels, adaDelta, 0.01, ModelCheckpoint(lr),
    ens::ProgressBar());

// Now files like model-1.bin, model-2.bin, etc. should be saved on disk.
```

---

Load an existing logistic regression model and print some information about it.

```c++
mlpack::LogisticRegression lr;
// This assumes that a model called "lr_model" has been saved to the file
// "model-1.bin" (as in the previous example).
mlpack::data::Load("model-1.bin", "lr_model", lr, true);

// Print the dimensionality of the model and some other statistics.
std::cout << "The dimensionality of the model in model-1.bin is "
    << (lr.Parameters().n_elem - 1) << "." << std::endl;
std::cout << "The bias parameter for the model is " << lr.Parameters()[0]
    << "." << std::endl;

arma::vec point(lr.Parameters().n_elem - 1, arma::fill::randu);
std::cout << "The predicted class for a random point, using a decision boundary"
    << " of 0.2, is " << lr.Classify(point, 0.2) << "." << std::endl;
```

---

Perform incremental training on multiple datasets with multiple calls to
`Train()`.

```c++
// Generate two random datasets.
arma::mat firstDataset(5, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> firstLabels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 1));

arma::mat secondDataset(5, 1500, arma::fill::randu); // 1500 points.
arma::Row<size_t> secondLabels =
    arma::randi<arma::Row<size_t>>(1500, arma::distr_param(0, 1));

// Train a model on the first dataset with an L2 regularization penalty
// parameter of 0.01.
mlpack::LogisticRegression lr(firstDataset, firstLabels, 0.01);

// Now compute the objective on the second dataset and print it.
std::cout << "Objective on second dataset: "
    << lr.ComputeError(secondDataset, secondLabels) << "." << std::endl;

// Train for a second round on the second dataset.
lr.Train(secondDataset, secondLabels);

// Now compute the objective on the second dataset again and print it.
// (Note that it may not be all that much better because this is random data!)
std::cout << "Objective on second dataset after second training: "
    << lr.ComputeError(secondDataset, secondLabels) << "." << std::endl;
```

---

### Advanced Functionality: Different Element Types

The `LogisticRegression` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```
LogisticRegression<MatType>
```

`MatType` specifies the type of matrix used for training data and internal
representation of model parameters.  Any matrix type that implements the
Armadillo API can be used.  The example below trains a logistic regression model
on sparse 32-bit floating point data.

```c++
// Create random, sparse 100-dimensional data.
arma::sp_fmat dataset;
dataset.sprandu(100, 5000, 0.3);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(5000, arma::distr_param(0, 1));

// Train with L2 regularization penalty parameter of 0.1.
mlpack::LogisticRegression<arma::sp_fmat> lr(dataset, labels, 0.1);

// Now classify a test point.
arma::sp_fvec point;
point.sprandu(100, 1, 0.3);

size_t prediction;
arma::fvec probabilitiesVec;
lr.Classify(point, prediction, probabilitiesVec);

std::cout << "Prediction for random test point: " << prediction << "."
    << std::endl;
std::cout << "Class probabilities for random test point: "
    << probabilitiesVec.t();
```

***Note***: if `MatType` is a sparse object (e.g. `sp_fmat`), the internal
parameter representation will be a *dense* vector containing elements of the
same type (e.g. `frowvec`).  This is because L2-regularized logistic regression,
even when training on sparse data, does not necessarily produce sparse models.
