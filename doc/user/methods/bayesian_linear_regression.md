## `BayesianLinearRegression`

The `BayesianLinearRegression` class implements a Bayesian ridge regression
model for numerical data that optimally tunes the regularization strength to the
given data.  The class offers configurable functionality and template parameters
to control the data type used for storing the model.

#### Simple usage example:

```c++
// Train a Bayesian linear regression model on random data and make predictions.

// All data and responses are uniform random; this uses 10 dimensional data.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::rowvec responses = arma::randn<arma::rowvec>(1000);
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::BayesianLinearRegression blr;  // Step 1: create model.
blr.Train(dataset, responses);         // Step 2: train model.
arma::rowvec predictions;
blr.Predict(testDataset, predictions); // Step 3: use model to predict.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 0.6) << " test points predicted to have"
    << " responses greater than 0.6." << std::endl;
std::cout << arma::accu(predictions < 0) << " test points predicted to have "
    << "negative responses." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `BayesianLinearRegression` objects.
 * [`Train()`](#training): train model.
 * [`Predict()`](#prediction): predict with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-different-element-types) for
   using different element types for a model.

#### See also:

 * [mlpack regression techniques](../modeling.md#regression)
 * [`LinearRegression`](linear_regression.md)
 * [`LARS`](lars.md)
 * [Bayesian linear regression on Wikipedia](https://en.wikipedia.org/wiki/Bayesian_linear_regression)

### Constructors

 * `blr = BayesianLinearRegression(centerData=true, scaleData=false, maxIterations=50, tolerance=1e-4)`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Predict()`](#prediction).

---

 * `blr = BayesianLinearRegression(data, responses)`
 * `blr = BayesianLinearRegression(data, responses, centerData=true, scaleData=false, maxIterations=50, tolerance=1e-4)`
   - Train model on the given data.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `responses` | [`arma::rowvec`](../matrices.md) | Training responses (e.g. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
| `centerData` | `bool` | Whether to center the data before learning. | `true` |
| `scaleData` | `bool` | Whether to scale the data to unit variance before learning. | `false` |
| `maxIterations` | `size_t` | Maximum number of iterations for convergence. | `50` |
| `tolerance` | `double` | Tolerance for convergence of the model. | `1e-4` |

As an alternative to passing `centerData`, `scaleData`, `maxIterations`, or
`tolerance`, they can each be set or accessed with standalone methods:

 * `blr.CenterData() = centerData;` will set whether to center the data before
   learning to `centerData`.
 * `blr.ScaleData() = scaleData;` will set whether to scale the data to unit
   variance before learning to `scaleData`.
 * `blr.MaxIterations() = maxIterations;` will set the maximum number of
   iterations to `maxIterations`.
 * `blr.Tolerance() = tolerance;` will set the tolerance for convergence to
   `tolerance`.

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

 * `blr.Train(data, responses, centerData=true, scaleData=false, maxIterations=50, tolerance=1e-4)`

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes:***

 * Training is not incremental.  A second call to `Train()` will retrain the
   model from scratch.

 * `Train()` returns the root mean squared error (RMSE) of the model on the
   training set as a `double`.

### Prediction

Once a `LinearRegression` model is trained, the `Predict()` member function
can be used to make predictions for new data.

 * `double predictedValue = blr.Predict(point)`
   - ***(Single-point)***
   - Make a prediction for a single point, returning the predicted value.

---
 * `blr.Predict(point, prediction, stddev)`
   - ***(Single-point)***
   - Make a prediction for a single point, storing the predicted value in
     `prediction` and the standard deviation of the prediction in `stddev`.

---

 * `blr.Predict(data, predictions)`
   - ***(Multi-point)***
   - Make predictions for a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

 * `blr.Predict(data, predictions, stddevs)`
   - ***(Multi-point)***
   - Make predictions for a set of points and compute standard deviations of
     predictions.
   - The prediction for data point `i` can be accessed with `predictions[i]`.
   - The standard deviation of the prediction for data point `i` can be accessed
     with `stddevs[i]`.

---

#### Prediction Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for prediction. |
| _single-point_ | `prediction` | `double&` | `double` to store predicted value into. |
| _single-point_ | `stddev` | `double&` | `double` to store standard deviation of predicted value into. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::rowvec&`](../matrices.md) | Vector of `double`s to store predictions into.  Will be set to length `data.n_cols`. |
| _multi-point_ | `stddevs` | [`arma::rowvec&`](../matrices.md) | Vector of `double`s to store standard deviations of predictions into.  Will be set to length `data.n_cols`. |

### Other Functionality

 * A `BayesianLinearRegression` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * After training is complete, the following methods can be used to inspect the
   model:
   - `blr.Omega()` returns the weights of the trained model as an
     `const arma::vec&` of length `data.n_rows`.  The weight for the `i`th
     dimension can be accessed with `blr.Omega()[i]`.

   - `blr.Alpha()` returns the precision (or inverse variance) of the Gaussian
     prior of the model as a `double`.

   - `blr.Beta()` returns the precision (or inverse variance) of the model as a
     `double`.

   - `blr.Variance()` returns the estimated variance as a `double`.

   - `blr.DataOffset()` returns a `const arma::vec&` containing the mean values
     of the training data in each dimension.  The vector has length
     `data.n_rows`.  The result is only meaningful if `centerData` is `true`.

   - `blr.DataScale()` returns a `const arma::vec&` containing the standard
     deviations of the training data in each dimension.  The vector has length
     `data.n_rows`.  The result is only meaningful if `scaleData` is `true`.

   - `blr.ResponsesOffset()` returns the mean value of the training responses as
     a `double`.  This is the intercept of the model.

 * `blr.RMSE(data, responses)` returns a `double` containing the RMSE (root mean
   squared error) of the model on the given `data` and `responses`.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `BayesianLinearRegression` class.

---

Train a Bayesian linear regression model in the constructor on weighted data,
compute the RMSE with `RMSE()`, and save the model.

```c++
// See https://datasets.mlpack.org/admission_predict.csv.
arma::mat data;
mlpack::data::Load("admission_predict.csv", data, true);

// See https://datasets.mlpack.org/admission_predict.responses.csv.
arma::rowvec responses;
mlpack::data::Load("admission_predict.responses.csv", responses, true);

// Generate random instance weights for each point, in the range 0.5 to 1.5.
arma::rowvec weights(data.n_cols, arma::fill::randu);
weights += 0.5;

// Train Bayesian linear regression model.  The data will be both centered and
// scaled to have unit variance.
mlpack::BayesianLinearRegression blr(data, responses, true, true);

// Now compute the RMSE on the training set.
std::cout << "RMSE on the training set: " << blr.RMSE(data, responses)
    << "." << std::endl;

// Finally, save the model with the name "blr".
mlpack::data::Save("blr_model.bin", "blr", blr, true);
```

---

Load a saved Bayesian linear regression model and print some information about
it, then make some predictions individually for random points.

```
mlpack::BayesianLinearRegression blr;

// Load the model named "blr" from "lr_model.bin".
mlpack::data::Load("blr_model.bin", "blr", blr, true);

// Print some information about the model.
const size_t dimensionality = blr.Omega().n_elem;
if (dimensionality == 0)
{
  std::cout << "The model in `blr_model.bin` has not been trained."
      << std::endl;
  return 0;
}

std::cout << "Information on the BayesianLinearRegression model in "
    << "'blr_model.bin':" << std::endl;
std::cout << " - Data was centered when training: "
    << (blr.CenterData() ? std::string("yes") : std::string("no")) << "."
    << std::endl;
std::cout << " - Data was scaled to unit variance when training: "
    << (blr.ScaleData() ? std::string("yes") : std::string("no")) << "."
    << std::endl;
std::cout << " - Model intercept: " << blr.ResponsesOffset() << "."
    << std::endl;
std::cout << " - Precision of Gaussian prior: " << blr.Alpha() << "."
    << std::endl;
std::cout << " - Precision of model: " << blr.Beta() << "." << std::endl;

// Now make a prediction for three random points.
for (size_t t = 0; t < 3; ++t)
{
  arma::vec randomPoint(dimensionality, arma::fill::randu);
  double prediction, stddev;
  blr.Predict(randomPoint, prediction, stddev);

  std::cout << "Prediction for random point " << t << ": " << prediction
      << " +/- " << stddev << "." << std::endl;
}
```

---

### Advanced Functionality: Different Element Types

The `BayesianLinearRegression` class has one template parameter that can be used
to control the element type of the model.  The full signature of the class is:

```
BayesianLinearRegression<ModelMatType>
```

`ModelMatType` specifies the type of matrix used for the internal representation
of model parameters.  Any matrix type that implements the Armadillo API can be
used; however, the matrix should be dense, as in general
`BayesianLinearRegression` will produce models that are not sparse.

The example below trains a Bayesian linear regression model on 32-bit floating
point data.

```c++
// Create random, sparse 100-dimensional data.
arma::fmat dataset(100, 5000, arma::fill::randu);

// Generate noisy responses from random data.
arma::fvec trueWeights(100, arma::fill::randu);
arma::frowvec responses = trueWeights.t() * dataset +
    0.01 * arma::randu<arma::frowvec>(5000) /* noise term */;

mlpack::BayesianLinearRegression<arma::fmat> blr;
blr.ScaleData() = true;
blr.MaxIterations() = 75;

blr.Train(dataset, responses);

// Compute the RMSE on the training set and a random test set.
arma::fmat testDataset(100, 1000, arma::fill::randu);

arma::frowvec testResponses = trueWeights.t() * testDataset +
    0.01 * arma::randu<arma::frowvec>(1000) /* noise term */;

std::cout << "RMSE on training set: "
    << blr.RMSE(dataset, responses) << "." << std::endl;
std::cout << "RMSE on test set:     "
    << blr.RMSE(testDataset, testResponses) << "." << std::endl;
```
