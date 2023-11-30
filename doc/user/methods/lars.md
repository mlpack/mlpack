## `LARS`


#### Simple usage example:

```c++
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

 * [`LinearRegression`](#linear_regression) <!-- TODO: fix link! -->
 * [mlpack regression techniques](#mlpack_regression_techniques) <!-- TODO: fix link! -->
 * [Least-angle Regression on Wikipedia](https://en.wikipedia.org/wiki/Least-angle_regression)

### Constructors

---

#### Constructor Parameters:

<!-- TODOs for table below:
    * better link for column-major matrices
    * update matrices.md to include a section on labels and NormalizeLabels()
 -->

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md) training matrix. | _(N/A)_ |
| `responses` | [`arma::rowvec`](../matrices.md) | Training responses (e.g. values to predict).  Should have length `data.n_cols`.  | _(N/A)_ |
| `weights` | [`arma::rowvec`](../matrices.md) | Weights for each training point.  Should have length `data.n_cols`. | _(N/A)_ |
| `lambda` | `double` | L2 regularization penalty parameter. | `0.0` |
| `intercept` | `bool` | Whether to fit an intercept term in the model. | `bool` |

### Training

If training is not done as part of the constructor call, it can be done with the
`Train()` function:

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes:***

### Prediction

Once a `LARS` model is trained, the `Predict()` member function
can be used to make predictions for new data.

 * `double predictedValue = lars.Predict(point)`
   - ***(Single-point)***
   - Make a prediction for a single point, returning the predicted value.

---

 * `lars.Predict(data, predictions)`
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

### Other Functionality

<!-- TODO: we should point directly to the documentation of those functions -->

 * A `LARS` model can be serialized with
   [`data::Save()`](../formats.md) and [`data::Load()`](../formats.md).

<!-- TODO: update for LARS -->
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
of the `LARS` class.

---

---

### Advanced Functionality: Different Element Types

The `LARS` class has one template parameter that can be used to
control the element type of the model.  The full signature of the class is:

```c++
LARS<ModelMatType>
```

`ModelMatType` specifies the type of matrix used for the internal representation
of model parameters.  Any matrix type that implements the Armadillo API can be
used.

Note that the `Train()` and `Predict()` functions themselves are templatized and
can allow any matrix type that has the same element type.  So, for instance, a
`LARS<arma::mat>` can accept an `arma::sp_mat` for training.

The example below TODO

```c++
```
