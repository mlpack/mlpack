## `LinearRegression`

The `LinearRegression` class implements a standard L2-regularized linear
regression model for numerical data, trained by direct decomposition of the
training data.  The class offers configurable functionality and template
parameters to control the data type used for storing the model.

#### Simple usage example:

```c++
// Train a linear regression model on random numeric data and make predictions.

// All data and responses are uniform random; this uses 10 dimensional data.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::rowvec responses = arma::randn<arma::rowvec>(1000);
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.

mlpack::LinearRegression lr;          // Step 1: create model.
lr.Train(dataset, responses);         // Step 2: train model.
arma::rowvec predictions;
lr.Predict(testDataset, predictions); // Step 3: use model to predict.

// Print some information about the test predictions.
std::cout << arma::accu(predictions > 0.7) << " test points predicted to have"
    << " responses greater than 0.7." << std::endl;
std::cout << arma::accu(predictions < 0) << " test points predicted to have "
    << "negative responses." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `LinearRegression` objects.
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
 * [`LARS`](lars.md)
 * [Linear Regression on Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)

### Constructors

 * `lr = LinearRegression()`
   - Initialize the model without training.
   - You will need to call [`Train()`](#training) later to train the model
     before calling [`Predict()`](#prediction).

---

 * `lr = LinearRegression(data, responses,          lambda=0.0, intercept=true)`
 * `lr = LinearRegression(data, responses, weights, lambda=0.0, intercept=true)`
   - Train model, optionally with instance weights.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `responses` | [`arma::rowvec`](../matrices.md) | Training responses (e.g. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`](../matrices.md) | Weights for each training point.  Should have length `data.n_cols`. | _(N/A)_ |
| `lambda` | `double` | L2 regularization penalty parameter. | `0.0` |
| `intercept` | `bool` | Whether to fit an intercept term in the model. | `bool` |

As an alternative to passing `lambda`, it can be set with the standalone
`Lambda()` method: `lr.Lambda() = l;` will set the value of `lambda` to `l` for
the next time `Train()` is called.

***Note***: setting `lambda` too small may cause the model to overfit; however,
setting it too large may cause the model to underfit.  [Automatic hyperparameter
tuning](../hpt.md) can be used to find a good value of `lambda` instead of a
manual setting.

<!-- TODO: update link to hyperparameter tuner -->

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

 * `lr.Train(data, responses,          lambda=0.0, intercept=true)`
 * `lr.Train(data, responses, weights, lambda=0.0, intercept=true)`
   - Train model on the given data, optionally with instance weights.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes:***

 * Training is not incremental.  A second call to `Train()` will retrain the
   model from scratch.

 * `Train()` returns the mean squared error (MSE) of the model on the training
   set as a `double`.

### Prediction

Once a `LinearRegression` model is trained, the `Predict()` member function
can be used to make predictions for new data.

 * `double predictedValue = lr.Predict(point)`
   - ***(Single-point)***
   - Make a prediction for a single point, returning the predicted value.

---

 * `lr.Predict(data, predictions)`
   - ***(Multi-point)***
   - Make predictions for a set of points.
   - The prediction for data point `i` can be accessed with `predictions[i]`.

---

#### Prediction Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for prediction. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictions` | [`arma::rowvec&`](../matrices.md) | Vector of `double`s to store predictions into.  Will be set to length `data.n_cols`. |

### Other Functionality

 * A `LinearRegression` model can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `lr.Intercept()` will return a `bool` indicating whether the model was
   trained with an intercept term.

 * `lr.Parameters()` will return an `arma::vec&` with the model parameters.
   This will have length equal to the dimensionality of the model if
   `lr.Intercept()` is `false`, and length equal to the dimensionality of the
   model plus one if `lr.Intercept()` is `true`.  If an intercept was fitted,
   the intercept term is the first element of `lr.Parameters()`.

 * `lr.ComputeError(data, responses)` will return a `double` containing the mean
   squared error (MSE) of the model on `data`, given that the true responses are
   `responses`.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `LinearRegression` class.

---

Train a linear regression model in the constructor on weighted data, compute the
objective function with `ComputeError()`, and save the model.

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

// Train a linear regression model, fitting an intercept term and using an L2
// regularization parameter of 0.3.
mlpack::LinearRegression lr(data, responses, weights, 0.3, true);

// Now compute the MSE on the training set.
std::cout << "MSE on the training set: " << lr.ComputeError(data, responses)
    << "." << std::endl;

// Finally, save the model with the name "lr".
mlpack::data::Save("lr_model.bin", "lr", lr, true);
```

---

Load a saved linear regression model and print some information about it, then
make some predictions individually for random points.

```
mlpack::LinearRegression lr;

// Load the model named "lr" from "lr_model.bin".
mlpack::data::Load("lr_model.bin", "lr", lr, true);

// Print some information about the model.
const size_t dimensionality =
    (lr.Intercept() ? (lr.Parameters().n_elem - 1) : lr.Parameters().n_elem);

std::cout << "Information on the LinearRegression model in 'lr_model.bin':"
    << std::endl;
std::cout << " - Model has intercept: "
    << (lr.Intercept() ? std::string("yes") : std::string("no")) << "."
    << std::endl;
if (lr.Intercept())
{
  std::cout << " - Intercept weight: " << lr.Parameters()[0] << "."
      << std::endl;
}
std::cout << " - Model dimensionality: " << dimensionality << "." << std::endl;
std::cout << " - Lambda value: " << lr.Lambda() << "." << std::endl;
std::cout << std::endl;

// Now make a prediction for three random points.
for (size_t t = 0; t < 3; ++t)
{
  arma::vec randomPoint(dimensionality, arma::fill::randu);
  const double prediction = lr.Predict(randomPoint);

  std::cout << "Prediction for random point " << t << ": " << prediction << "."
      << std::endl;
}
```

---

See also the following fully-working examples:

 - [Salary prediction with `LinearRegression`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/linear_regression/salary_prediction/salary-prediction-cpp.ipynb)
 - [Avocado price prediction with `LinearRegression`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/linear_regression/avocado_price_prediction/avocado_price_prediction_cpp.ipynb)
 - [California housing price prediction with `LinearRegression`](https://github.com/mlpack/examples/blob/master/jupyter_notebook/linear_regression/california_housing_price_prediction/california_housing_price_prediction_cpp.ipynb)

### Advanced Functionality: Different Element Types

The `LinearRegression` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```
LinearRegression<ModelMatType>
```

`ModelMatType` specifies the type of matrix used for the internal representation
of model parameters.  Any matrix type that implements the Armadillo API can be
used.

Note that the `Train()` and `Predict()` functions themselves are templatized and
can allow any matrix type that has the same element type.  So, for instance, a
`LinearRegression<arma::mat>` can accept an `arma::sp_mat` for training.

The example below trains a linear regression model on sparse 32-bit floating
point data, but uses a dense 32-bit floating point vector to store the model
itself.

```c++
// Create random, sparse 100-dimensional data.
arma::sp_fmat dataset;
dataset.sprandu(100, 5000, 0.3);

// Generate noisy responses from random data.
arma::fvec trueWeights(100, arma::fill::randu);
arma::frowvec responses = trueWeights.t() * dataset +
    0.01 * arma::randu<arma::frowvec>(5000) /* noise term */;

mlpack::LinearRegression<arma::fmat> lr;
lr.Lambda() = 0.01;

lr.Train(dataset, responses);

// Compute the MSE on the training set and a random test set.
arma::sp_fmat testDataset;
testDataset.sprandu(100, 1000, 0.3);

arma::frowvec testResponses = trueWeights.t() * testDataset +
    0.01 * arma::randu<arma::frowvec>(1000) /* noise term */;

std::cout << "MSE on training set: "
    << lr.ComputeError(dataset, responses) << "." << std::endl;
std::cout << "MSE on test set:     "
    << lr.ComputeError(testDataset, testResponses) << "." << std::endl;
```

***Note:*** dense objects should be used for `ModelMatType`, since in general an
L2-regularized linear regression model will not be sparse.
