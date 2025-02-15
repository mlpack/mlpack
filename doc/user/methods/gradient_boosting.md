## `GradBoosting`

The `GradBoosting` class implements a gradient boosting model that supports
numerical and categorical features. The class offers several template 
parameters and several runtime options that can be used to control the 
model.

Gradient Boosting is a very powerful ensemble algorithm used for both 
classification and regression tasks. It utilizes a series of weak learners 
(specifically [decision tree regressors](decision_tree_regressor.md)) to arrive 
closer and closer to a targeted label. Each 
weak learner is trained on the error of the previous learner, thereby 
reducing the error of the overall model. `GradBoosting` is useful for classifying points with _discrete labels_ (i.e. `0`,
`1`, `2`).

#### Simple usage example:

```c++
// Train a gradient boosting model on random numeric data and predict labels 
// on test data: All data and labels are uniform random; 10 dimensional data, 
// 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.
size_t numWeakLearners = 10;
size_t numClasses = 5;

mlpack::GradBoosting gb;                                // Step 1: create model.
gb.Train(dataset, labels, numClasses, numWeakLearners); // Step 2: train model.
arma::Row<size_t> predictions;
gb.Classify(testDataset, predictions);                  // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `GradientBoosting` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.

#### See also:

 * [`AdaBoost`](adaboost.md)
 * [`DecisionTree`](decision_tree.md)
 * [`DecisionTreeRegressor`](decision_tree_regressor.md)
 * [Random forests](random_forest.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [Gradient Boosting on Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)

### Constructors

 * `gb = GradBoosting()`
   - Initialize model without training.
   - You will need to call [`Train()`](#training) later to train the model before
     calling [`Classify()`](#classification).

---

 * `gb = GradBoosting(data, datasetInfo, labels, numClasses, numWeakLearners)`
   - Train on numerical and/or categorical data using default arguments for the
     weak learners ([`DecisionTreeRegressor`s](decision_tree_regressor.md)).

---

 * `gb = GradBoosting(data, datasetInfo, labels, numClasses, numWeakLearners, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical and/or categorical data, using the given hyperparameters.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](https://github.com/mlpack/mlpack/blob/master/doc/user/load_save.md#loading-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../load_save.md#normalizing-labels) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `numWeakLearners` | `size_t` | Number of weak learner models to be used. | 5 |
| `minimumLeafSize` | `size_t` | Minimum leaf size for weak learner decision tree. | `10` |
| `minimumGainSplit` | `double` | Minimum gain split for weak learner decision tree | `1e-7` |
| `maximumDepth` | `size_t` | Maximum tree depth for weak learner decision tree | `2` |
 * Smaller values of `numWeakLearners` and `maximumDepth` will result in smaller `GradientBoosting` models, but the model may underfit the data.

  * Smaller values of `minimumLeafSize` and `minimumGainSplit` will improve the performance of the `GradientBoosting` model, but at the cost of training time and model size.

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `gb.Train(data, datasetInfo, labels, numClasses, numWeakLearners)`
   - Train on numerical data without using any decision tree arguments.

---

 * `gb.Train(data, datasetInfo, labels, numClasses, numWeakLearners, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed categorical data using the `minimumLeafSize`, `minimumGainSplit`, and `maximumDepth` options to control the behavior of the weak learners ([`DecisionTreeRegressor`s](decision_tree_regressor.md)).

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Note:***: Training is not incremental.  A second call to `Train()` will retrain the
model from scratch.

### Classification

Once a `GradBoosting` object is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t predictedClass = gb.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `gb.Classify(point, prediction)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.

---

 * `gb.Classify(data, predictedLabels)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The prediction for data point `i` can be accessed with `predictedLabels[i]`.

---

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictedLabels` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into.  Will be set to length `data.n_cols`. |

***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other Functionality

 * `WeakLearner(size_t i)` returns a [`DecisionTreeRegressor&`]
    (decision_tree_regressor.md) indicating which weak learner 
    is being used. 

 * `gb.NumWeakLearners()` returns a `size_t` indicating how 
    many weak learners are being used by the model.
    - `gb.NumWeakLearners(x)` will set the number of weak learners 
    that will be used the *next time* `Train()` is called to `x`.

 * `gb.NumClasses()` returns a `size_t` indicating the number of classes the
   model was trained on.

For complete functionality, the [source
code](/src/mlpack/methods/grad_boosting/grad_boosting.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`GradBoosting`.

---

Train a gradient boosting model on mixed categorical data, passing 
arguments into the Train function:

```c++
// Load a categorical dataset.
arma::mat dataset;
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, true);

// Create the model object.
mlpack::GradBoosting gb;
// Train on the given dataset, specifying the number of weak learners as 5.
gb.Train(dataset, info, labels, 7 /* classes */, 5 /* number of weak learners */);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, true);

// Predict class of first test point.
const size_t firstPrediction = gb.Classify(testDataset.col(0));
std::cout << "Predicted class of first test point is " << firstPrediction << "."
    << std::endl;

```

---

Train a gradient boosting model on mixed categorical data, passing 
arguments into the Constructor function:

```c++
// Load a categorical dataset.
arma::mat dataset;
mlpack::data::DatasetInfo info;
// See https://datasets.mlpack.org/covertype.train.arff.
mlpack::data::Load("covertype.train.arff", dataset, info, true);

arma::Row<size_t> labels;
// See https://datasets.mlpack.org/covertype.train.labels.csv.
mlpack::data::Load("covertype.train.labels.csv", labels, true);

// Create the model object.
mlpack::GradBoosting gb(dataset, info, labels, 
   7 /* classes */, 5 /* number of weak learners */);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, true);

// Predict class of first test point.
const size_t firstPrediction = gb.Classify(testDataset.col(0));
std::cout << "Predicted class of first test point is " << firstPrediction << "."
    << std::endl;

```

---
