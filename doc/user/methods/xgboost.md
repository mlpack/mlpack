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
// Train a XGBoost model on the Iris dataset and predict labels 
// on test data: 
// Replace with a data::Load() call or similar for a real application.
// numModels is a hyperparameter refering to the number of weak learners

arma::mat db;
mlpack::data::DatasetInfo info;
if (!data::Load("iris_train.csv", db, info))
FAIL("Cannot load test dataset iris_train.csv!");

arma::Row<size_t> labels;
if (!data::Load("iris_train_labels.csv", labels))
FAIL("Cannot load labels for iris iris_train_labels.txt");

arma::mat testDb;
if (!data::Load("iris_test.csv", testDb))
FAIL("Cannot load test dataset iris_test.csv!");

arma::Row<size_t> testLabels;
if (!data::Load("iris_test_labels.csv", testLabels))
FAIL("Cannot load test dataset iris_test_labels.csv!");

const size_t numClasses = arma::max(labels.row(0)) + 1;
const size_t numModels = 5;

mlpack::XGBoost xgb;               // Step 1: create model.
xgb.Train(db, labels, info, numClasses, numModels);          // Step 2: train model.
arma::Row<size_t> predictions;
xgb.Classify(testDb, predictions); // Step 3: classify points.

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

 * `xgb = XGBoost(data, labels, datasetInfo, numClasses, numModels)`
   - Train on mixed category data using default arguments for the weak learners (Decision Trees).

---

 * `xgb = XGBoost(data, labels, datasetInfo, numClasses, numModels, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed category data, entering the weak learner arguments `minimumLeafSize`, `minimumGainSplit` and `maximumDepth`.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. | _(N/A)_ |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels. [between `0` and `numClasses - 1`](../load_save.md#normalizing-labels) (inclusive).  Should have length `data.n_cols`.  | _(N/A)_ |
| `datasetInfo` | [`data::DatasetInfo`](https://github.com/mlpack/mlpack/blob/master/doc/user/load_save.md#loading-categorical-data) | Dataset information, specifying type information for each dimension. | _(N/A)_ |
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

 * `xgb.Train(data, labels, datasetInfo, numClasses, numModels)`
   - Train on mixed category data without using any decision tree arguments.

---

 * `xgb.Train(data, labels, datasetInfo, numClasses, numModels, minimumLeafSize, minimumGainSplit, maximumDepth)`
   - Train on mixed category data using weak learner model's arguments `minimumLeafSize`, `minimumGainSplit` and `maximumDepth`.

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
    - Classify a single point.
    - The predicted class is stored in `prediction`.

---

 * `xgb.Classify(point, prediction, probabilities)`
    - ***(Single-point)***
    - Classify a single point and compute class probabilities.
    - The predicted class is stored in `prediction`.
    - Class probabilities are stored in `probabilities`.


---

 * `xgb.Classify(data, predictedLabels)`
    - ***(Multi-point)***
    - Classify a set of points.
    - The prediction for data point `i` can be accessed with `predictedLabels[i]`.

---

 * `xgb.Classify(data, predictedLabels, probabilities)`
    - ***(Multi-point)***
    - Classify a set of points and compute class probabilities.
    - The prediction for data point `i` can be accessed with `predictedLabels[i]`.
    - Class probabilities are stored in `probabilities`.

---

#### Classification Parameters:

| **usage** | **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| _single-point_ | `point` | [`arma::vec`](../matrices.md) | Single point for classification. |
| _single-point_ | `prediction` | `size_t&` | `size_t` to store class prediction into. |
| _single-point_ | `probabilities` | [`arma::rowvec`](../matrices.md) | Single row to store probabilities, size equal to numClasses |
||||
| _multi-point_ | `data` | [`arma::mat`](../matrices.md) | Set of [column-major](../matrices.md#representing-data-in-mlpack) points for classification. |
| _multi-point_ | `predictedLabels` | [`arma::Row<size_t>&`](../matrices.md) | Vector of `size_t`s to store class prediction into.  Will be set to length `data.n_cols`. |
| _multi-point_ | `probabilities` | [`arma::mat`](../matrices.md) | Matrice to store probabilities, dimensions equal to numClasses x data.n_cols |


***Note:*** different types can be used for `data` and `point` (e.g.
`arma::fmat`, `arma::sp_mat`, `arma::sp_vec`, etc.).  However, the element type
that is used should be the same type that was used for training.

### Other Functionality

 * `xgb.NumModels()` returns a `size_t` indicating how many weak learners are being
    used by the model.

 * `xgb.NumClasses()` returns a `size_t` indicating the number of classes the
   model was trained on.

 * `xgb.SetNumClasses(size_t x)` set the number of classes to `x` explicitly.

 * `xgb.SetNumModels(size_t x)` set the number of weak learners to `x` explicitly.

For complete functionality, the [source
code](/src/mlpack/methods/xgboost/xgboost.hpp) can be consulted.
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
xgb.Train(dataset, labels, info, 7 /* classes */, 5 /* number of weak learners */);

// Load categorical test data.
arma::mat testDataset;
// See https://datasets.mlpack.org/covertype.test.arff.
mlpack::data::Load("covertype.test.arff", testDataset, info, true);

// Predict class of first test point.
const size_t firstPrediction = xgb.Classify(testDataset.col(0));
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
