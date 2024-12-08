## `LARS`

The `LARS` class implements the least-angle regression (LARS) algorithm for
L1-penalized and L2-penalized linear regression.  `LARS` can also solve the
LASSO (least absolute shrinkage and selection operator) problem.  The LARS
algorithm is a *path* algorithm, and thus will recover solutions for *all* L1
penalty parameters greater than or equal to the given L1 penalty parameter.

#### Simple usage example:

```c++
// Train a LARS model on random numeric data and make predictions.

// All data and responses are uniform random; this uses 10 dimensional data.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::rowvec responses = arma::randn<arma::rowvec>(1000);
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::LARS lars(true, 0.1 /* L1 penalty */); // Step 1: create model.
lars.Train(dataset, responses);                // Step 2: train model.
arma::rowvec predictions;
lars.Predict(testDataset, predictions);        // Step 3: use model to predict.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 0.7) << " test points predicted to have"
    << " responses greater than 0.7." << std::endl;
std::cout << arma::accu(predictions < 0) << " test points predicted to have "
    << "negative responses." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `LARS` objects.
 * [`Train()`](#training): train model.
 * [`Predict()`](#prediction): predict with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [The LARS path](#the-lars-path): use models from the LARS path with different
   L1 penalty values.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-different-element-types) for
   using different element types for a model.

#### See also:

 * [`LinearRegression`](linear_regression.md)
 * [mlpack regression techniques](../modeling.md#regression)
 * [Least-angle Regression on Wikipedia](https://en.wikipedia.org/wiki/Least-angle_regression)

### Constructors

 * `lars = LARS(useCholesky=false, lambda1=0.0, lambda2=0.0, tolerance=1e-16, fitIntercept=true, normalizeData=true)`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Predict()`](#prediction).

---

 * `lars = LARS(data, responses, colMajor=true, useCholesky=true, lambda1=0.0, lambda2=0.0, tolerance=1e-16, fitIntercept=true, normalizeData=true)`
   - Train model on the given data and responses, using the given settings for
     hyperparameters.

---

 * `lars = LARS(data, responses, colMajor, useCholesky, gramMatrix, lambda1=0.0, lambda2=0.0, tolerance=1e-16, fitIntercept=true, normalizeData=true)`
   - *(Advanced constructor)*.
   - Train model on the given data and responses, using a precomputed Gram
     matrix (`gramMatrix`, equivalent to `data * data.t()`).
   - Using a precomputed Gram matrix can save time, if it has already been
     computed.
   - ***Note:*** any precomputed Gram matrix must also match the settings of
     `fitIntercept` and `normalizeData`; so, if both are `true`, then
     `gramMatrix` must be computed on mean-centered data whose features are
     normalized to have unit variance.  In addition, if `lambda2 > 0`, then
     it is expected that `lambda2` is added to each element on the diagonal of
     `gramMatrix`.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | Training matrix. | _(N/A)_ |
| `responses` | [`arma::rowvec`](../matrices.md) | Training responses (e.g. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
| `colMajor` | `bool` | Should be set to `true` if `data` is [column-major](../matrices.md#representing-data-in-mlpack).  Passing row-major data can avoid a transpose operation. | `false` |
| `useCholesky` | `bool` | If `true`, use the Cholesky decomposition of the Gram matrix to solve linear systems (as opposed to the full Gram matrix). | `false` |
| `gramMatrix` | [`arma::mat`](../matrices.md) | Precomputed Gram matrix of `data` (i.e.  `data * data.t()` for column-major data). | _(N/A)_ |
| `lambda1` | `double` | L1 regularization penalty parameter. | `0.0` |
| `lambda2` | `double` | L2 regularization penalty parameter. | `0.0` |
| `tolerance` | `double` | Tolerance on feature correlations for convergence. | `1e-16` |
| `fitIntercept` | `bool` | If `true`, an intercept term will be included in the model. | `true` |
| `normalizeData` | `bool` | If `true`, data will be normalized before fitting the model. | `true` |

As an alternative to passing hyperparameters, each hyperparameter can be set
with a standalone method.  The following functions can be used before calling
`Train()` to set hyperparameters:

 * `lars.UseCholesky() = useChol;` will set whether or not the Cholesky
   decomposition will be used during training to `useChol`.
 * `lars.Lambda1() = lambda1;` will set the L1 regularization penalty parameter
   to `lambda1`.
 * `lars.Lambda2() = lambda2;` will set the L2 regularization penalty parameter
   to `lambda2`.
 * `lars.Tolerance() = tol;` will set the convergence tolerance to `tol`.
 * `lars.FitIntercept(fitIntercept);` will set whether an intercept will be fit
   to `fitIntercept`.  If an external Gram matrix has been specified, this will
   throw an exception.
 * `lars.NormalizeData(normalizeData);` will set whether data should be
   normalized to `normalizeData`.  If an external Gram matrix has been
   specified, this will throw an exception.

***Notes:***

 - The `lambda1` parameter implicitly controls the sparsity of the model; for
   more sparse models (i.e. fewer nonzero weights), specify a larger `lambda1`.

 - Specifying a too-small `lambda1` or `lambda2` value may cause the model to
   overfit; however, setting it too large may cause the model to underfit.
   Because LARS is a path algorithm, the
   [`SelectBeta()`](#other-functionality) functions can be used to select models
   with different values of `lambda1`.  For tuning `lambda2`,
   [Automatic hyperparameter tuning](../hpt.md) can be used.

<!-- TODO: fix the link to the hyperparameter tuner -->

 - `fitIntercept` and `normalizeData` are recommended to be set as `true`, in
   accordance with the original LARS algorithm.  `false` can be used for
   `fitIntercept` if the features and responses are already mean-centered, and
   `false` can also be used for `normalizeData` if the features are already
   unit-variance.  Using `false` for either option can provide a small amount of
   speedup.

 - `useCholesky` should generally be set to `true` and in most situations will
   result in faster training.

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

 * `lars.Train(data, responses, colMajor=true, useCholesky=true, lambda1=0.0, lambda2=0.0, tolerance=1e-16, fitIntercept=true, normalizeData=true)`
   - Train the model on the given data.

---

 * `lars.Train(data, responses, colMajor, useCholesky, gramMatrix, lambda1=0.0, lambda2=0.0, tolerance=1e-16, fitIntercept=true, normalizeData=true)`
   - *(Advanced training.)*
   - Train model on the given data and responses, using a precomputed Gram
     matrix (`gramMatrix`, equivalent to `data * data.t()`).
   - Using a precomputed Gram matrix can save time, if it has already been
     computed.
   - ***Note:*** any precomputed Gram matrix must also match the settings of
     `fitIntercept` and `normalizeData`; so, if both are `true`, then
     `gramMatrix` must be computed on mean-centered data whose features are
     normalized to have unit variance.  In addition, if `lambda2 > 0`, then
     it is expected that `lambda2` is added to each element on the diagonal of
     `gramMatrix`.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes:***

 * Training is not incremental.  A second call to `Train()` will retrain the
   model from scratch.

 * `Train()` returns the squared error (loss) of the model on the training set
   as a `double`.  To obtain the MSE, divide by the number of training points.

### Prediction

Once a `LARS` model is trained, the `Predict()` member function
can be used to make predictions for new data.

 * `double predictedValue = lars.Predict(point)`
   - ***(Single-point)***
   - Make a prediction for a single point, returning the predicted value.

---

 * `lars.Predict(data, predictions, colMajor=true)`
   - ***(Multi-point)***
   - Make predictions for a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

#### Prediction Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for prediction. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md) points for classification. |
| _multi-point_ | `predictions` | [`arma::rowvec&`](../matrices.md) | Vector of `double`s to store predictions into.  Will be set to length `data.n_cols`. |
| _multi-point_ | `colMajor` | `bool` | Should be set to `true` if `data` is [column-major](../matrices.md).  Passing row-major data can avoid a transpose operation.  (Default `true`.) |

### Other Functionality

 * A `LARS` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `lars.Beta()` will return an `arma::vec` with the model parameters.  This
   will have length equal to the dimensionality of the model.  Note that
   `lars.Beta()` can be changed to a different model on the LARS path using the
   [`lars.SelectBeta()` method](#the-lars-path).

 * `lars.Intercept()` will return a `double` representing the fitted intercept
   term, or 0 if `lars.FitIntercept()` is `false`.

 * `lars.ActiveSet()` will return a `std::vector<size_t>&` containing the
   indices of nonzero dimensions in the model parameters (`lars.Beta()`).

 * `lars.ComputeError(data, responses, colMajor=true)` will return a `double`
   containing the squared error of the model on `data`, given that
   the true responses are `responses`.  To obtain the MSE, divide by the number
   of points in `data`.

### The LARS Path

LARS is a *path* (or stepwise) algorithm, meaning it adds one feature at a time
to the model.  This in turn means that when we train a LARS model with
`lambda1` set to `l`, we also recover every possible LARS model on the same data
with a `lambda1` greater than `l`.

The `LARS` class provides a way to access all of the models on the path, and
switch between them for prediction purposes:

 * `lars.BetaPath()` returns a `std::vector<arma::vec>&` containing each set of
   model weights on the LARS path.

 * `lars.InterceptPath()` returns a `std::vector<double>&` containing each
   intercept value on the LARS path.  These values are only meaningful if
   `lars.FitIntercept()` is `true`.

 * `lars.LambdaPath()` returns a `std::vector<double>&` containing each
   `lambda1` value that is associated with each element in `lars.BetaPath()` and
   `lars.InterceptPath()`.  That is, `lars.LambdaPath()[i]` is the `lambda1`
    value corresponding to the model defined by `lars.BetaPath()[i]` and
    `lars.InterceptPath()[i]`.

 * `lars.SelectBeta(lambda1)` will set the model weights (`lars.ActiveSet()`,
   `lars.Beta()` and `lars.Intercept()`) to the path location with L1 penalty
   `lambda1`.  This is equivalent to calling `lars.Train(data, responses,
   colMajor, useCholesky, lambda1)`---but much more efficient!  `lambda1`
   cannot be less than `lars.Lambda1()`, or an exception will be thrown.

 * `lars.SelectedLambda1()` returns the currently selected L1 regularization
   penalty parameter.

 * For any value `lambda1` between `lars.LambdaPath()[i]` and
   `lars.LambdaPath()[i + 1]`, the corresponding model is a linear interpolation
   between `lars.BetaPath()[i]` and `lars.BetaPath()[i + 1]` (and
   `lars.InterceptPath()[i]` and `lars.InterceptPath()[i + 1]`).  This exact
   linear interpolation is what is computed by `lars.SelectBeta(lambda1)`.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `LARS` class.

---

Train a LARS model in the constructor, and print the MSE on training and
test data for each set of weights in the path.

```c++
// See https://datasets.mlpack.org/wave_energy_farm_100.csv.
arma::mat data;
mlpack::data::Load("wave_energy_farm_100.csv", data, true);

// Split the last row off: it is the responses.  Also, normalize the responses
// to [0, 1].
arma::rowvec responses = data.row(data.n_rows - 1);
responses /= responses.max();
data.shed_row(data.n_rows - 1);

// Split into a training and test dataset.  20% of the data is held out as a
// test set.
arma::mat trainingData, testData;
arma::rowvec trainingResponses, testResponses;
mlpack::data::Split(data, responses, trainingData, testData, trainingResponses,
    testResponses, 0.2);

// Train a LARS model with lambda1 = 1e-5 and lambda2 = 1e-6.
mlpack::LARS lars(trainingData, trainingResponses, true, true, 1e-5, 1e-6);

// Iterate over all the models in the path.
const size_t pathLength = lars.BetaPath().size();
for (size_t i = 0; i < pathLength; ++i)
{
  // Use the i'th model in the path.
  lars.SelectBeta(lars.LambdaPath()[i]);

  // ComputeError() returns the total loss, which we need to divide by the
  // number of points to get the MSE.
  const double trainMSE = lars.ComputeError(trainingData, trainingResponses) /
      trainingData.n_cols;
  const double testMSE = lars.ComputeError(testData, testResponses) /
      testData.n_cols;
  std::cout << "L1 penalty parameter: " << lars.SelectedLambda1() << std::endl;
  std::cout << "  MSE on training set: " << trainMSE << "." << std::endl;
  std::cout << "  MSE on test set:     " << testMSE << "." << std::endl;
}
```

---

Train a LARS model, print predictions for a random point, and save to a file.

```c++
// See https://datasets.mlpack.org/admission_predict.csv.
arma::mat data;
mlpack::data::Load("admission_predict.csv", data, true); 

// See https://datasets.mlpack.org/admission_predict.responses.csv.
arma::rowvec responses;
mlpack::data::Load("admission_predict.responses.csv", responses, true);

// Train a LARS model with only L2 regularization.
mlpack::LARS lars(data, responses, true, true, 0.0, 0.1 /* lambda2 */);

// Predict on a random point.
arma::vec point = arma::randu<arma::vec>(data.n_rows);
const double prediction = lars.Predict(point);

std::cout << "Prediction on random point: " << prediction << "." << std::endl;

// Save the model to "lars_model.bin" with the name "lars".
mlpack::data::Save("lars_model.bin", "lars", lars, true);
```

---

Load a LARS model from disk and print some information about it.

```c++
// This assumes a model named "lars" has previously been saved to
// "lars_model.bin".
mlpack::LARS lars;
mlpack::data::Load("lars_model.bin", "lars", lars, true);

if (lars.BetaPath().size() == 0)
{
  std::cout << "lars_model.bin contains an untrained LARS model." << std::endl;
}
else
{
  std::cout << "Information on the LARS model in lars_model.bin:" << std::endl;

  std::cout << " - Model dimensionality: " << lars.Beta().n_elem << "."
      << std::endl;
  std::cout << " - Has intercept: "
      << (lars.FitIntercept() ? std::string("yes") : std::string("no")) << "."
      << std::endl;
  std::cout << " - Current L1 regularization penalty parameter value: "
      << lars.SelectedLambda1() << "." << std::endl;
  std::cout << " - L2 regularization penalty parameter: " << lars.Lambda2()
      << "." << std::endl;
  std::cout << " - Number of nonzero elements in model: "
      << lars.ActiveSet().size() << "." << std::endl;
  std::cout << " - Number of models in LARS path: " << lars.BetaPath().size()
      << "." << std::endl;
  std::cout << " - Model weight for dimension 0: " << lars.Beta()[0] << "."
      << std::endl;

  if (lars.FitIntercept())
  {
    std::cout << " - Intercept value: " << lars.Intercept() << "." << std::endl;
  }
}
```

---

Train several models with different L2 regularization penalty parameters, using
a precomputed Gram matrix.

```c++
// See https://datasets.mlpack.org/admission_predict.csv.
arma::mat data;
mlpack::data::Load("admission_predict.csv", data, true);

// See https://datasets.mlpack.org/admission_predict.responses.csv.
arma::rowvec responses;
mlpack::data::Load("admission_predict.responses.csv", responses, true);

// Precompute Gram matrix.
arma::mat gramMatrix = data * data.t();

std::vector<double> lambda2Values = { 0.01, 0.1, 1.0, 10.0, 100.0 };
for (double lambda2 : lambda2Values)
{
  // Build a LARS model using the precomputed Gram matrix.  We did not normalize
  // or center the data before computing the Gram matrix, so we have to set
  // fitIntercept and normalizeData accordingly.
  mlpack::LARS lars(data, responses, true, true, gramMatrix, 0.01, lambda2,
      1e-16, false, false);

  std::cout << "MSE with L2 penalty " << lambda2 << ": "
      << (lars.ComputeError(data, responses) / data.n_cols) << "." << std::endl;
}
```

### Advanced Functionality: Different Element Types

The `LARS` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```
LARS<ModelMatType>
```

`ModelMatType` specifies the type of matrix used for the internal representation
of model parameters.  Any matrix type that implements the Armadillo API can be
used.

Note that the `Train()` and `Predict()` functions themselves are templatized and
can allow any matrix type that has the same element type.  So, for instance, a
`LARS<arma::sp_mat>` can accept an `arma::mat` for training.

The example below trains a LARS model on 32-bit precision data, using
`arma::sp_fmat` to store the model parameters.

```c++
// Create random, sparse 1000-dimensional data.
arma::fmat dataset(1000, 5000, arma::fill::randu);

// Generate noisy responses from random data.
arma::fvec trueWeights(1000, arma::fill::randu);
arma::frowvec responses = trueWeights.t() * dataset +
    0.01 * arma::randu<arma::frowvec>(5000) /* noise term */;

mlpack::LARS<arma::sp_fmat> lars;
lars.Lambda1() = 0.1;
lars.Lambda2() = 0.01;

lars.Train(dataset, responses);

// Compute the MSE on the training set and a random test set.
arma::fmat testDataset(1000, 2500, arma::fill::randu);
arma::frowvec testResponses = trueWeights.t() * testDataset +
    0.01 * arma::randu<arma::frowvec>(2500) /* noise term */;

const float trainMSE = lars.ComputeError(dataset, responses) / dataset.n_cols;
const float testMSE = lars.ComputeError(testDataset, testResponses) /
    testDataset.n_cols;

std::cout << "MSE on training set: " << trainMSE << "." << std::endl;
std::cout << "MSE on test set:     " << testMSE << "." << std::endl;
```

***Note:*** it is generally only more efficient to use a sparse type (e.g.
`arma::sp_mat`) for `ModelMatType` when the L1 regularization parameter is set
such that a highly sparse model is produced.
