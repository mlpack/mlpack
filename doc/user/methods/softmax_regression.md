## `SoftmaxRegression`

The `SoftmaxRegression` class implements an L2-regularized multi-class softmax
regression classifier for numerical data.  This is a multi-class extension of
the logistic regression classifier.  By default, L-BFGS is used to learn the
model.  The `SoftmaxRegression` class offers easy configurability, and arbitrary
optimizers can be used to learn the model.

Softmax regression is useful for multi-class classification (i.e. classes are
`0`, `1`, `2`).  For two-class situations, see also the
[`LogisticRegression`](logistic_regression.md) class.

#### Simple usage example:

```c++
// Train a softmax regression model on random data and predict labels:

// All data and labels are uniform random; 5 dimensional data, 4 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(5, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 3));
arma::mat testDataset(5, 500, arma::fill::randu); // 500 test points.

mlpack::SoftmaxRegression sr;          // Step 1: create model.
sr.Train(dataset, labels, 4);          // Step 2: train model.
arma::Row<size_t> predictions;
sr.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `SoftmaxRegression` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-different-element-types) for
   using different element types for a model.

#### See also:

 * [`LogisticRegression`](logistic_regression.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [UFLDL Softmax Regression Tutorial](http://deeplearning.stanford.edu/tutorial/supervised/SoftmaxRegression/)

### Constructors

 * `sr = SoftmaxRegression()`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Classify()`](#classification).

---

 * `sr = SoftmaxRegression(data, labels, numClasses, lambda=0.0001, fitIntercept=true)`
 * `sr = SoftmaxRegression(data, labels, numClasses, lambda=0.0001, fitIntercept=true, [callbacks...])`
   - Train model, optionally specifying hyperparameters and callbacks for the
     default ensmallen optimizer.

---

 * `sr = SoftmaxRegression(data, labels, numClasses, optimizer, lambda=0.0001, fitIntercept=true)`
 * `sr = SoftmaxRegression(data, labels, numClasses, optimizer, lambda=0.0001, fitIntercept=true, [callbacks...])`
    - Train model with a custom ensmallen optimizer, optionally specifying
      hyperparameters and callbacks for the custom optimizer.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `lambda` | `double` | L2 regularization penalty parameter.  Must be nonnegative. | `0.0001` |
| `fitIntercept` | `bool` | If true, an intercept term is fit to the model. | `true` |
| `optimizer` | [any ensmallen optimizer](https://www.ensmallen.org) | Instantiated ensmallen optimizer for [differentiable functions](https://www.ensmallen.org/docs.html#differentiable-functions) or [differentiable separable functions](https://www.ensmallen.org/docs.html#differentiable-separable-functions). | _(N/A)_ |
| `callbacks...` | [any set of ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation) | Optional callbacks for the ensmallen optimizer, such as e.g. `ens::ProgressBar()`, `ens::Report()`, or others. | _(N/A)_ |

As an alternative to passing `lambda`, it can be set with a
standalone method:

 * `sr.Lambda() = l;` will set the value of the L2 regularization penalty
   parameter to `l`.

It is not possible to set `fitIntercept` except in the constructor or the call
to `Train()`.

***Note***: Setting `lambda` too small may cause the model to overfit; however,
setting it too large may cause the model to underfit.  [Automatic hyperparameter
tuning](../hpt.md) can be used to find a good value of `lambda` instead of a
manual setting.

<!-- TODO: fix hyperparameter tuner link with real documentation -->

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

 * `sr.Train(data, labels, numClasses, lambda=0.0001, fitIntercept=true)`
 * `sr.Train(data, labels, numClasses, lambda=0.0001, fitIntercept=true, [callbacks...])`
   - Train model, optionally specifying hyperparameters and callbacks for the
     default ensmallen optimizer.

---

 * `sr.Train(data, labels, numClasses, optimizer, lambda=0.0001, fitIntercept=true)`
 * `sr.Train(data, labels, numClasses, optimizer, lambda=0.0001, fitIntercept=true, optimizer=ens::L_BFGS(), [callbacks...])`
   - Train model with a custom ensmallen optimizer, optionally specifying
    hyperparameters and callbacks for the optimizer.

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

 * `Train()` returns a `double` with the final softmax regression loss value
   (including L2 penalty term) of the trained model.

### Classification

Once a `SoftmaxRegression` model is trained, the `Classify()` member function
can be used to make class predictions for new data.

 * `size_t predictedClass = sr.Classify(point)`
   - ***(Single-point)***
   - Classify a single point, returning the predicted class (`0` through
     `numClasses - 1`, inclusive).

---

 * `sr.Classify(point, prediction, probabilitiesVec)`
   - ***(Single-point)***
   - Classify a single point and compute class probabilities.
   - The predicted class is stored in `prediction`.
   - The probability of class `i` can be accessed with `probabilitiesVec[i]`.

---

 * `sr.Classify(data, predictions)`
   - ***(Multi-point)***
   - Classify a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `sr.Classify(data, predictions, probabilities)`
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

 * A `SoftmaxRegression` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `sr.Parameters()` will return an `arma::mat` filled with the weights of the
   model.  The matrix will have rows equal to `numClasses` and columns equal to
   the dimensionality of the model (plus one if `fitIntercept` is `true`).  Each
   row of the matrix can be considered its own two-class one-vs.-all logistic
   regression model.

 * `sr.NumClasses()` will return the number of classes the model was trained on.

 * `sr.ComputeAccuracy(data, labels)` will return the accuracy of the model on
   the given `data` with the given `labels`.  The returned accuracy is between 0
   and 100.

 * `sr.Reset()` will reset the weights of the model to small random values.

For complete functionality, the [source
code](/src/mlpack/methods/softmax_regression/softmax_regression.hpp) can be
consulted.  Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `SoftmaxRegression` class.

---

Train a softmax regression model using a custom SGD-like optimizer with
callbacks.

```c++
// See https://datasets.mlpack.org/mnist.train.csv.
arma::mat dataset;
mlpack::data::Load("mnist.train.csv", dataset, true);
// See https://datasets.mlpack.org/mnist.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("mnist.train.labels.csv", labels, true);

mlpack::SoftmaxRegression sr;

// Create AdaGrad optimizer with custom step size and batch size.
ens::AdaGrad optimizer(0.0005 /* step size */, 16 /* batch size */);
optimizer.MaxIterations() = 10 * dataset.n_cols; // Allow 10 epochs.

// Print a progress bar and an optimization report when training is finished.
sr.Train(dataset, labels, 10 /* numClasses */, optimizer,
    0.01 /* lambda */, true /* fit intercept */, ens::ProgressBar(),
    ens::Report());

// Now predict on test labels and compute accuracy.

// See https://datasets.mlpack.org/mnist.test.csv.
arma::mat testDataset;
mlpack::data::Load("mnist.test.csv", testDataset, true);
// See https://datasets.mlpack.org/mnist.test.labels.csv.
arma::Row<size_t> testLabels;
mlpack::data::Load("mnist.test.labels.csv", testLabels, true);

std::cout << std::endl;
std::cout << "Accuracy on training set: "
    << sr.ComputeAccuracy(dataset, labels) << "\%." << std::endl;
std::cout << "Accuracy on test set:     "
    << sr.ComputeAccuracy(testDataset, testLabels) << "\%." << std::endl;
```

---

Train a softmax regression model with AdaGrad and save the model every epoch
using a [custom ensmallen
callback](https://www.ensmallen.org/docs.html#custom-callbacks):

```c++
// This callback saves the model into "model-<epoch>.bin" after every epoch.
class ModelCheckpoint
{
 public:
  ModelCheckpoint(mlpack::SoftmaxRegression<>& model) : model(model) { }

  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* optimizer */,
                FunctionType& /* function */,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double /* objective */)
  {
    const std::string filename = "model-" + std::to_string(epoch) + ".bin";
    mlpack::data::Save(filename, "sr_model", model, true);
    return false; // Do not terminate the optimization.
  }

 private:
  mlpack::SoftmaxRegression<>& model;
};
```

With that callback available, the code to train the model is below:

```c++
// See https://datasets.mlpack.org/mnist.train.csv.
arma::mat dataset;
mlpack::data::Load("mnist.train.csv", dataset, true);
// See https://datasets.mlpack.org/mnist.train.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("mnist.train.labels.csv", labels, true);

mlpack::SoftmaxRegression sr;

// Create AdaGrad optimizer with small step size and batch size of 32.
ens::AdaGrad adaGrad(0.0005, 32);
adaGrad.MaxIterations() = 10 * dataset.n_cols; // 10 epochs maximum.

// Use the custom callback and an L2 penalty parameter of 0.01.
sr.Train(dataset, labels, 10 /* numClasses */, adaGrad, 0.01, true,
    ModelCheckpoint(sr), ens::ProgressBar());

// Now files like model-1.bin, model-2.bin, etc. should be saved on disk.
```

---

Load an existing softmax regression model and print some information about it.

```c++
mlpack::SoftmaxRegression sr;
// This assumes that a model called "sr_model" has been saved to the file
// "model-1.bin" (as in the previous example).
mlpack::data::Load("model-1.bin", "sr_model", sr, true);

// Print the dimensionality of the model and some other statistics.
const size_t dimensionality = (sr.FitIntercept()) ?
    (sr.Parameters().n_cols - 1) : (sr.Parameters().n_cols);
std::cout << "The dimensionality of the model in model-1.bin is "
    << dimensionality << "." << std::endl;
std::cout << "The bias parameters for the model, for each class, are: "
    << std::endl;
std::cout << sr.Parameters().col(0).t();

arma::vec point(dimensionality, arma::fill::randu);
std::cout << "The predicted class for a random point is " << sr.Classify(point)
    << "." << std::endl;
```

---

Perform incremental training on multiple datasets with multiple calls to
`Train()`.

```c++
// Generate two random datasets with four classes.
arma::mat firstDataset(5, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> firstLabels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 3));

arma::mat secondDataset(5, 1500, arma::fill::randu); // 1500 points.
arma::Row<size_t> secondLabels =
    arma::randi<arma::Row<size_t>>(1500, arma::distr_param(0, 3));

// Train a model on the first dataset with an L2 regularization penalty
// parameter of 0.01, not fitting an intercept.
mlpack::SoftmaxRegression sr(firstDataset, firstLabels, 4, 0.01, false);

// Now compute the accuracy on the second dataset and print it.
std::cout << "Accuracy on second dataset: "
    << sr.ComputeAccuracy(secondDataset, secondLabels) << "\%." << std::endl;

// Train for a second round on the second dataset.
sr.Train(secondDataset, secondLabels, 4);

// Now compute the accuracy on the second dataset again and print it.
// (Note that it may not be all that much better because this is random data!)
std::cout << "Accuracy on second dataset after second training: "
    << sr.ComputeAccuracy(secondDataset, secondLabels) << "\%." << std::endl;
```

---

### Advanced Functionality: Different Element Types

The `SoftmaxRegression` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```
SoftmaxRegression<MatType>
```

`MatType` specifies the type of matrix used for training data and internal
representation of model parameters.  Any matrix type that implements the
Armadillo API can be used.  The example below trains a softmax regression model
on sparse 32-bit floating point data.

```c++
// Create random, sparse 100-dimensional data, with 3 classes.
arma::sp_fmat dataset;
dataset.sprandu(100, 5000, 0.3);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(5000, arma::distr_param(0, 2));

// Train with L2 regularization penalty parameter of 0.1.
mlpack::SoftmaxRegression<arma::sp_fmat> sr(dataset, labels, 3, 0.1);

// Now classify a test point.
arma::sp_fvec point;
point.sprandu(100, 1, 0.3);

size_t prediction;
arma::fvec probabilitiesVec;
sr.Classify(point, prediction, probabilitiesVec);

std::cout << "Prediction for random test point: " << prediction << "."
    << std::endl;
std::cout << "Class probabilities for random test point: "
    << probabilitiesVec.t();
```

***Note***: if `MatType` is a sparse object (e.g. `sp_fmat`), the internal
parameter representation will be a *dense* matrix containing elements of the
same type (e.g. `fmat`).  This is because L2-regularized softmax regression,
even when training on sparse data, does not necessarily produce sparse models.
