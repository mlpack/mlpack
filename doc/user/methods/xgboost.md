## `XGBoost`

The `XGBoost` class implements a XGBoost model that supports
numerical and categorical features. The class offers several template 
parameters and several runtime options that can be used to control the 
model.

XGBoost (eXtreme Gradient Boosting) is a very powerful ensemble algorithm developed
by Tianqi Chen. It is built based on the Gradient Boosting algorithm, with 
additional features. XGBoost is used for both classification and regression tasks. 
It utilises a series of weak learners (eg. Decision Stumps) to arrive closer and closer 
to a targeted label. Each weak learner is trained on the error of the 
previous learner, thereby reducing the error of the overall model.

#### Simple usage example:

```c++
// Train a XGBoost model on random numeric data and predict labels 
// on test data: All data and labels are uniform random; 10 dimensional data, 
// 5 classes.
// Replace with a data::Load() call or similar for a real application.
// numModels is a hyperparameter refering to the number of weak learners
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));
arma::mat testDataset(10, 500, arma::fill::randu); // 500 test points.
size_t numModels = 10;
size_t numClasses = 5;

mlpack::XGBoost xgb;               // Step 1: create model.
xgb.Train(dataset, labels, numClasses, numModels);          // Step 2: train model.
arma::Row<size_t> predictions;
xgb.Classify(testDataset, predictions); // Step 3: classify points.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 2) << " test points classified as class "
    << "2." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `XGBoost` objects.
 * [`Train()`](#training): train model.
 * [`Classify()`](#classification): classify with a trained model.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.

#### See also:

 * [`Gradient Boosting`](gradient_boosting.md)
 * [`DecisionTreeRegressor`](decision_tree_regressor.md)
 * [Random forests](random_forest.md)
 * [mlpack classifiers](../../index.md#classification-algorithms)
 * [XGBoost on Wikipedia](https://en.wikipedia.org/wiki/XGBoost)

### Constructors

 * `xgb = XGBoost()`
   - Initialize model without training.
   - You will need to call [`Train()`](#training) later to train the model before
     calling [`Classify()`](#classification).

---

 * `xgb = XGBoost(data, labels, numClasses, numModels)`
   - Train on numerical-only data using default arguments for the weak learners (Decision Trees).

---

 * `xgb = XGBoost(data, labels, numClasses, numModels, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical-only data, entering the weak learner arguments `minimumLeafSize`, `minimumGainSplit` and `maximumDepth`.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels. [between `0` and `numClasses - 1`](../load_save.md#normalizing-labels) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `numClasses` | `size_t` | Number of classes in the dataset. | _(N/A)_ |
| `numModels` | `size_t` | Number of weak learner models to be used. | _(N/A)_ |
| `minimumLeafSize` | `size_t` | Minimum leaf size for weak learner decision tree | 10 |
| `minimumGainSplit` | `double` | Minimum gain split for weak learner decision tree | 1e-7 |
| `maximumDepth` | `size_t` | Maximum tree depth for weak learner decision tree | 2 |

 * Number of labels in the data must be less than or equal to numClasses.

***Note:*** different types can be used for `data` (e.g.,
`arma::fmat`, `arma::sp_mat`). 

### Training

If training is not done as part of the constructor call, it can be done with one
of the following versions of the `Train()` member function:

 * `xgb.Train(data, labels, numClasses, numModels)`
   - Train on numerical data without using any decision tree arguments.

---

 * `xgb.Train(data, labels, numClasses, numModels, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on numerical data using weak learner model's arguments `minimumLeafSize`, `minimumGainSplit` and `maximumDepth`.

---

Types of each argument are the same as in the table for constructors
[above](#constructor-parameters).

***Notes***:

 * Training is not incremental.  A second call to `Train()` will retrain the
   model from scratch.

### Classification

Once a `XGBoost` is trained, the `Classify()` member function can be used
to make class predictions for new data.

 * `size_t predictedClass = xgb.Classify(point)`
    - ***(Single-point)***
    - Classify a single point, returning the predicted class.

---

 * `xgb.Classify(point, prediction)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.

---

 * `xgb.Classify(data, predictedLabels)`
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

 * `WeakLearner(size_t i)` returns a `WeakLearnerType` indicating which weak learner 
    is being used. 

 * `xgb.NumModels()` returns a `size_t` indicating how many weak learners are being
    used by the model.

 * `xgb.NumClasses()` returns a `size_t` indicating the number of classes the
   model was trained on.

 * `xgb.SetNumModels(size_t x)` set the number of weak learners to `x` explicitly.

For complete functionality, the [source
code](/src/mlpack/methods/grad_boosting/grad_boosting.hpp) can be consulted.
Each method is fully documented.

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial use of
`XGBoost`.

---

Train a XGBoost model on mixed categorical data and save it:

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
mlpack::XGBoost xgb;
// Train on the given dataset, specifying number of weak learners at 5.
xgb.Train(dataset, labels, 7 /* classes */, 5 /* number of weak learners */);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, info, true);

// Predict class of first test point.
const size_t firstPrediction = gb.Classify(testDataset.col(0));
std::cout << "Predicted class of first test point is " << firstPrediction << "."
    << std::endl;

// Save the model to `xgb.bin`.
mlpack::data::Save("xgb.bin", "xgb", xgb);
```

---

Load a model and print some information about it.

```c++
mlpack::XGBoost xgb;
// This call assumes a xgb called "xgb" has already been saved to `xgb.bin`
// with `data::Save()`.
mlpack::data::Load("xgb.bin", "xgb", xgb, true);

std::cout << "The number of weak learners being used by the model is "
<< xgb.NumModels() << "." << std::endl;

```

---
