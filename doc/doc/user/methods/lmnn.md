## LMNN

The `LMNN` class implements large margin nearest neighbor, which can be used
as both a linear dimensionality reduction technique and a distance learning
technique (also called metric learning).  LMNN finds a linear transformation of
the dataset that improves `k`-nearest-neighbor classification performance.

#### Simple usage example:

```c++
// Learn a distance metric that improves kNN classification performance.

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

mlpack::LMNN lmnn(3 /* neighbors to consider */); // Step 1: create object.
arma::mat distance;
lmnn.LearnDistance(dataset, labels, distance);    // Step 2: learn distance.

// `distance` can now be used as a transformation matrix for the data.
arma::mat transformedData = distance * dataset;
// Or, you can create a MahalanobisDistance to evaluate points in the
// transformed dataset space.
arma::mat q = distance.t() * distance;
mlpack::MahalanobisDistance d(std::move(q));

std::cout << "Distance between points 0 and 1:" << std::endl;
std::cout << " - Before LMNN: "
    << mlpack::EuclideanDistance::Evaluate(dataset.col(0), dataset.col(1))
    << "." << std::endl;
std::cout << " - After LMNN:  "
    << d.Evaluate(dataset.col(0), dataset.col(1)) << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `LMNN` objects.
 * [`LearnDistance()`](#learning-distances): learn distance metrics.
 * [Other functionality](#other-functionality) for loading and saving.
 * [Examples](#simple-examples) of simple usage and integration with other
   techniques.

#### See also:

<!-- TODO: link to kNN -->

 * [mlpack distance metrics](../core/distances.md)
 * [`NCA`](nca.md)
 * [Metric learning on Wikipedia](https://en.wikipedia.org/wiki/Similarity_learning#Metric_learning)
 * [Large margin nearest neighbor on Wikipedia](https://en.wikipedia.org/wiki/Large_margin_nearest_neighbor)
 * [Distance metric learning for Large Margin Nearest Neighbor Classification (pdf)](https://proceedings.neurips.cc/paper_files/paper/2005/file/a7f592cef8b130a6967a90617db5681b-Paper.pdf)

### Constructors

 * `lmnn = LMNN(k, regularization=0.5, updateInterval=1)`
   - Create an `LMNN` object considering the specified number `k` of neighbors.
   - Optionally, specify the regularization to be applied to the LMNN cost
     function (a `double`), and the number of iterations between recomputation
     of neighbors (`updateInterval`, a `size_t`).

---

 * `lmnn = LMNN<DistanceType>(k, regularization=0.5, updateInterval=1)`
 * `lmnn = LMNN<DistanceType>(k, regularization,     updateInterval,   distance)`
   - Create an `LMNN` object using a custom
     [`DistanceType`](../core/distances.md).
   - `k` specifies the number of neighbors to consider.
   - `regularization` specifies the regularization penalty to be applied to the
     LMNN cost function (a `double`).
   - `updateInterval` specifies the number of iterations between recomputation
     of neighbors (a `size_t`).
   - An instantiated `DistanceType` can optionally be passed with the `distance`
     parameter.
   - Using a custom `DistanceType` means that `LearnDistance()` will learn a
     linear transformation for the data *in the metric space of the custom
     `DistanceType`*.
     * This means any learned distance may not necessarily improve
       classification performance with the
       [Euclidean distance](../core/distances.md#lmetric).
     * Instead, classification performance will be improved when the learned
       distance is used with the given `DistanceType` only.
   - Any mlpack `DistanceType` can be used as a drop-in replacement, or a
     [custom `DistanceType`](../../developer/distances.md).
     * A list of mlpack's provided distance metrics can be found
       [here](../core/distances.md).
   - ***Note: be sure that you understand the implications of a custom
     `DistanceType` before using this version.***

---

***Notes***:

 - A larger `k` will cause `LearnDistance()` to take longer to compute, but will
   give more accurate results.  It is generally suggested to keep `k` in roughly
   the `3` to `5` range, depending on the dataset.  Using `k = 1` can provide
   fast convergence, but the learned distance metric may be of lower quality.

 - `regularization` controls the balance between encouraging small distances for
   points of the same class and penalizing small distances for points of
   different classes.  When `regularization` is increased, small distances for
   points of different classes are further penalized.

 - Setting `updateInterval` greater than `1` will allow the LMNN algorithm to
   take multiple steps without the expensive recomputation of neighbors, but
   this means that subsequent optimization steps may not be using the true
   nearest neighbors.
   * If using an SGD-like algorithm (i.e. an optimizer for a
     [differentiable separable function](https://www.ensmallen.org/docs.html#differentiable-separable-functions)),
     this can often be set to a relatively high value (100 is not unreasonable).
   * If using an optimizer like L-BFGS (i.e. a full-batch optimizer for
     [differentiable functions](https://www.ensmallen.org/docs.html#differentiable-functions)),
     this should be kept relatively low (going above 10 is not advised).
   * It is worth cross-validating different values of the parameter to see what
     works for your dataset.

---

### Learning Distances

Once an `LMNN` object has been created, the `LearnDistance()` method can be used
to learn a distance.

 * `lmnn.LearnDistance(data, labels, distance,            [callbacks...])`
 * `lmnn.LearnDistance(data, labels, distance, optimizer, [callbacks...])`
   - Learn a distance metric on the given `data` and `labels`, filling
     `distance` with a transformation matrix that can be used to map the data
     into the space of the learned distance.
   - Optionally, pass an instantiated
     [ensmallen optimizer](https://www.ensmallen.org) and/or
     [ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation)
     to be used for the learning process.
   - If no optimizer is passed,
     [`ens::AMSGrad`](https://www.ensmallen.org/docs.html#amsgrad) is used.
   - If `distance` already has size `r` x `data.n_rows` for some `r` less than
     or equal to `data.n_rows`, it will be used as the starting point for
     optimization.  Otherwise, the identity matrix with size `data.n_rows` x
     `data.n_rows` will be used.
   - When optimization is complete, `distance` will have size `r` x
     `data.n_rows`, where `r` is less than or equal to `data.n_rows`.
     * *Note*: If `r < data.n_rows`, then LMNN has learned a distance metric
       that also reduces the dimensionality of the data.  See the
       [last example](#simple-examples).

To use `distance`, either:

 * Compute a new transformed dataset as `distance * data`, or
 * Use an instantiated
   [`MahalanobisDistance`](../core/distances.md#mahalanobisdistance)
   with `distance.t() * distance` as the `Q` matrix.

See the [examples section](#simple-examples) for more details.

#### `LearnDistance()` Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  |
| `distance` | [`arma::mat`](../matrices.md) | Output matrix to store transformation matrix representing learned distance. |
| `optimizer` | [any ensmallen optimizer](https://www.ensmallen.org) | Instantiated ensmallen optimizer for [differentiable functions](https://www.ensmallen.org/docs.html#differentiable-functions) or [differentiable separable functions](https://www.ensmallen.org/docs.html#differentiable-separable-functions). | `ens::AMSGrad()` |
| `callbacks...` | [any set of ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation) | Optional callbacks for the ensmallen optimizer, such as e.g. `ens::ProgressBar()`, `ens::Report()`, or others. | _(N/A)_ |

***Note***: any matrix type can be used for `data` and `distance`, so long as
that type implements the Armadillo API.  So, e.g., `arma::fmat` can be used.

### Other Functionality

 * An `LMNN` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).
   Note that this is only meaningful if a custom `DistanceType` is being used,
   and that custom `DistanceType` has state to be saved.

 * `lmnn.K()` returns the number of neighbors used by LMNN, and `lmnn.K() = k`
   will set the number of neighbors to use to `k`.

 * `lmnn.Regularization()` returns the current regularization value of the LMNN
   object (as a `double`), and `lmnn.Regularization() = r` can be used to set
   the regularization value to `r`.

 * `lmnn.UpdateInterval()` returns the current number of iterations between
   neighbor recomputation (as a `size_t`), and `lmnn.UpdateInterval() = i` sets
   the number of iterations between neighbor recomputation to `i`.

 * `lmnn.Distance()` will return the `DistanceType` being used for learning.
   Unless a custom `DistanceType` was specified in the constructor,
   this simply returns a
   [`SquaredEuclideanDistance`](../core/distances.md#lmetric) object.

### Simple Examples

Learn a distance metric to improve classification performance on the iris
dataset, and show improved performance when using
[`NaiveBayesClassifier`](naive_bayes_classifier.md).

```c++
// See https://datasets.mlpack.org/satellite.test.csv.
// (We are using the test set here just because it is a little smaller and
// we want this example to run quickly.)
arma::mat dataset;
mlpack::data::Load("satellite.test.csv", dataset, true);
// See https://datasets.mlpack.org/satellite.test.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.test.labels.csv", labels, true);

// Create an LMNN object using 5 nearest neighbors and learn a distance.
arma::mat distance;
mlpack::LMNN lmnn(5);
lmnn.LearnDistance(dataset, labels, distance);

// The distance matrix has size equal to the dimensionality of the data.
std::cout << "Learned distance size: " << distance.n_rows << " x "
    << distance.n_cols << "." << std::endl;

// Learn a NaiveBayesClassifier model on the data and print the performance.
mlpack::NaiveBayesClassifier nbc1(dataset, labels, 2);
arma::Row<size_t> predictions;
nbc1.Classify(dataset, predictions);
std::cout << "Naive Bayes Classifier without LMNN: "
    << arma::accu(labels == predictions) << " of " << labels.n_elem
    << " correct." << std::endl;

// Now transform the data and learn another NaiveBayesClassifier.
arma::mat transformedDataset = distance * dataset;
mlpack::NaiveBayesClassifier nbc2(transformedDataset, labels, 2);
nbc2.Classify(transformedDataset, predictions);
std::cout << "Naive Bayes Classifier with LMNN:    "
    << arma::accu(labels == predictions) << " of " << labels.n_elem
    << " correct." << std::endl;
```

---

Learn a distance metric on the vehicle dataset, using 32-bit floating point to
represent the data and metric.

```c++
// See https://datasets.mlpack.org/vehicle.csv.
arma::fmat dataset;
mlpack::data::Load("vehicle.csv", dataset, true);

// The labels are contained as the last row of the dataset.
arma::Row<size_t> labels =
    arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
dataset.shed_row(dataset.n_rows - 1);

// Create an LMNN object with k=1 and learn distance on float32 data.
// Set updateInterval to a large value (100) because we are using the default
// AMSGrad optimizer (which will take very many small steps).
arma::fmat distance;
mlpack::LMNN lmnn(1, 0.5, 100);

lmnn.LearnDistance(dataset, labels, distance, ens::ProgressBar());

// We want to compute six quantities:
//
//  - Average distance to points of the same class before LMNN.
//  - Average distance to points of the same class after LMNN, using
//    MahalanobisDistance.
//  - Average distance to points of the same class after LMNN, using the
//    transformed dataset.
//
//  - The same three quantities above, but for points of the other class.
//
// LMNN should reduce the average distance to points in the same class, while
// increasing the average distance to points in other classes.
float distSums[6] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
size_t sameCount = 0;
arma::fmat q = distance.t() * distance;
mlpack::MahalanobisDistance md(std::move(q));
arma::fmat transformedDataset = distance * dataset;
for (size_t i = 1; i < dataset.n_cols; ++i)
{
  const double d1 = mlpack::EuclideanDistance::Evaluate(
      dataset.col(0), dataset.col(i));
  const double d2 = md.Evaluate(dataset.col(0), dataset.col(i));
  const double d3 = mlpack::EuclideanDistance::Evaluate(
      transformedDataset.col(0), transformedDataset.col(i));

  // Determine whether the point has the same label as point 0.
  if (labels[i] == labels[0])
  {
    distSums[0] += d1;
    distSums[1] += d2;
    distSums[2] += d3;
    ++sameCount;
  }
  else
  {
    distSums[3] += d1;
    distSums[4] += d2;
    distSums[5] += d3;
  }
}

// Turn the results into average distances across the class.
distSums[0] /= sameCount;
distSums[1] /= sameCount;
distSums[2] /= sameCount;
distSums[3] /= (dataset.n_cols - sameCount);
distSums[4] /= (dataset.n_cols - sameCount);
distSums[5] /= (dataset.n_cols - sameCount);

// Print the results.
std::cout << "Average distance between point 0 and other points of the same "
    << "class:" << std::endl;
std::cout << " - Before LMNN:                           " << distSums[0] << "."
    << std::endl;
std::cout << " - After LMNN (with MahalanobisDistance): " << distSums[1] << "."
    << std::endl;
std::cout << " - After LMNN (with transformed dataset): " << distSums[2] << "."
    << std::endl;
std::cout << std::endl;

std::cout << "Average distance between point 0 and points of other classes: "
    << std::endl;
std::cout << " - Before LMNN:                           " << distSums[3] << "."
    << std::endl;
std::cout << " - After LMNN (with MahalanobisDistance): " << distSums[4] << "."
    << std::endl;
std::cout << " - After LMNN (with transformed dataset): " << distSums[5] << "."
    << std::endl;
std::cout << std::endl;

std::cout << "Ratio of other-class to same-class distances:" << std::endl;
std::cout << "(We expect this to go up.)" << std::endl;
std::cout << " - Before LMNN: " << (distSums[3] / distSums[0]) << "."
    << std::endl;
std::cout << " - After LMNN:  " << (distSums[5] / distSums[2]) << "."
    << std::endl;
```

---

Learn a distance metric on the iris dataset, using the L-BFGS optimizer with
callbacks.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("iris.labels.csv", labels, true);

// Learn a distance with ensmallen's L-BFGS optimizer.
ens::L_BFGS lbfgs;
lbfgs.NumBasis() = 5;
lbfgs.MaxIterations() = 1000;

// Use 5 neighbors for LMNN, and leave updateInterval at the default of 1,
// because we are using L-BFGS (a full-back optimizer).
mlpack::LMNN lmnn(5);

// Use a callback that prints a final optimization report.
arma::mat distance;
lmnn.LearnDistance(dataset, labels, distance, lbfgs, ens::Report());
```

---

Learn a distance metric on the vehicle dataset, but instead of using the
Euclidean distance as the underlying metric, use the Manhattan distance.  This
means that LMNN is optimizing k-NN performance under the Manhattan distance, not
under the Euclidean distance.

```c++
// See https://datasets.mlpack.org/vehicle.csv.
arma::mat dataset;
mlpack::data::Load("vehicle.csv", dataset, true);

// The labels are contained as the last row of the dataset.
arma::Row<size_t> labels =
    arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
dataset.shed_row(dataset.n_rows - 1);

// Create the LMNN object and optimize.  Use k=3 and Nesterov momentum SGD,
// printing a progress bar during optimization.  Because Nesterov momentum SGD
// is an ensmallen optimizer for differentiable separable functions, we increase
// updateInterval to reduce the number of neighbor recomputations.  We also set
// the regularization parameter to 1.0 to increase the penalty for nearby
// neighbors of a different class.
mlpack::LMNN<mlpack::ManhattanDistance> lmnn(3, 1.0, 100);
arma::mat distance;
ens::NesterovMomentumSGD opt(0.000001 /* step size */,
                             32 /* batch size */,
                             20 * dataset.n_cols /* 20 epochs */);
lmnn.LearnDistance(dataset, labels, distance, opt, ens::ProgressBar());

// Now inspect distances between points with the Euclidean distance and with the
// inner product distance.
arma::mat transformedDataset = distance * dataset;

// Points 0 and 1 have the same label (0).  See their original distance---with
// both the Euclidean and Manhattan distances---and their transformed distances.
// We expect these points to get closer together, in the Manhattan distance.
const double d1 = mlpack::ManhattanDistance::Evaluate(
    dataset.col(0), dataset.col(1));
const double d2 = mlpack::ManhattanDistance::Evaluate(
    transformedDataset.col(0), transformedDataset.col(1));

std::cout << "Distance between points 0 and 1 (same class):" << std::endl;
std::cout << " - Manhattan distance:" << std::endl;
std::cout << "   * Before LMNN: " << d1 << std::endl;
std::cout << "   * After LMNN:  " << d2 << std::endl;
std::cout << std::endl;

// Point 3 has a different label.  We therefore expect this point to get further
// from point 0 with the Manhattan distance, but not necessarily with the
// Euclidean distance.
const double d3 = mlpack::ManhattanDistance::Evaluate(
    dataset.col(0), dataset.col(3));
const double d4 = mlpack::ManhattanDistance::Evaluate(
    transformedDataset.col(0), transformedDataset.col(3));

std::cout << "Distance between points 0 and 3 (different class):" << std::endl;
std::cout << " - Manhattan distance:" << std::endl;
std::cout << "   * Before LMNN: " << d3 << std::endl;
std::cout << "   * After LMNN:  " << d4 << std::endl;

// Note that point 3 has been moved further away from point 0 than point 1.
```

---

Learn a distance metric while also performing dimensionality reduction, reducing
the dimensionality of the satellite dataset by 3 dimensions.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, true);
// See https://datasets.mlpack.org/satellite.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("satellite.train.labels.csv", labels, true);

// Use a random initialization for the distance transformation, with the
// specified output dimensionality.
arma::mat distance(dataset.n_rows - 3, dataset.n_rows, arma::fill::randu);
mlpack::LMNN lmnn(3);
ens::L_BFGS opt;
opt.MaxIterations() = 10; // You may want more in a real application.
lmnn.LearnDistance(dataset, labels, distance, opt, ens::Report());

// Now transform the dataset.
arma::mat transformedData = distance * dataset;

std::cout << "Original data has size " << dataset.n_rows << " x "
    << dataset.n_cols << "." << std::endl;
std::cout << "Transformed data has size " << transformedData.n_rows << " x "
    << transformedData.n_cols << "." << std::endl;
```
