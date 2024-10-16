## NCA

The `NCA` class implements neighborhood components analysis, which can be used
as both a linear dimensionality reduction technique and a distance learning
technique (also called metric learning).  Neighborhood components analysis finds
a linear transformation of the dataset that improves `k`-nearest-neighbor
classification performance.

Note that `NCA` is a computationally intensive technique (each optimization
iteration takes time quadratic in the data size!), and may be slow to run even
for datasets of only moderate size.  See [`LMNN`](lmnn.md) for another distance
learning technique that scales better to larger datasets.

#### Simple usage example:

```c++
// Learn a distance metric that improves kNN classification performance.

// All data and labels are uniform random; 10 dimensional data, 5 classes.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

mlpack::NCA nca;                              // Step 1: create object.
arma::mat distance;
nca.LearnDistance(dataset, labels, distance); // Step 2: learn distance.

// `distance` can now be used as a transformation matrix for the data.
arma::mat transformedData = distance * dataset;
// Or, you can create a MahalanobisDistance to evaluate points in the
// transformed dataset space.
arma::mat q = distance.t() * distance;
mlpack::MahalanobisDistance d(std::move(q));

std::cout << "Distance between points 0 and 1:" << std::endl;
std::cout << " - Before NCA: "
    << mlpack::EuclideanDistance::Evaluate(dataset.col(0), dataset.col(1))
    << "." << std::endl;
std::cout << " - After NCA:  "
    << d.Evaluate(dataset.col(0), dataset.col(1)) << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `NCA` objects.
 * [`LearnDistance()`](#learning-distances): learn distance metrics.
 * [Other functionality](#other-functionality) for loading and saving.
 * [Examples](#simple-examples) of simple usage and integration with other
   techniques.

#### See also:

<!-- TODO: link to kNN -->

 * [mlpack distance metrics](../core/distances.md)
 * [`LMNN`](lmnn.md)
 * [Metric learning on Wikipedia](https://en.wikipedia.org/wiki/Similarity_learning#Metric_learning)
 * [Neighborhood Components Analysis on Wikipedia](https://en.wikipedia.org/wiki/Neighbourhood_components_analysis)
 * [Neighbourhood Components Analysis (pdf)](https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf)

### Constructors

 * `nca = NCA()`
   - Create an `NCA` object with default parameters.

---

 * `nca = NCA<DistanceType>()`
 * `nca = NCA<DistanceType>(distance)`
   - Create an `NCA` object using a custom
     [`DistanceType`](../core/distances.md).
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

### Learning Distances

Once an `NCA` object has been created, the `LearnDistance()` method can be used
to learn a distance.

 * `nca.LearnDistance(data, labels, distance,            [callbacks...])`
 * `nca.LearnDistance(data, labels, distance, optimizer, [callbacks...])`
   - Learn a distance metric on the given `data` and `labels`, filling
     `distance` with a transformation matrix that can be used to map the data
     into the space of the learned distance.
   - Optionally, pass an instantiated
     [ensmallen optimizer](https://www.ensmallen.org) and/or
     [ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation)
     to be used for the learning process.
   - If `distance` already has size `r` x `data.n_rows` for some `r` less than
     or equal to `data.n_rows`, it will be used as the starting point for
     optimization.  Otherwise, the identity matrix with size `data.n_rows` x
     `data.n_rows` will be used.
   - When optimization is complete, `distance` will have size `r` x
     `data.n_rows`, where `r` is less than or equal to `data.n_rows`.
     * *Note*: If `r < data.n_rows`, then NCA has learned a distance metric that
       also reduces the dimensionality of the data.  See the
       [last example](#simple-examples).

To use `distance`, either:

 * Compute a new transformed dataset as `distance * data`, or
 * Use an instantiated
   [`MahalanobisDistance`](../core/distances.md#mahalanobisdistance)
   with `distance.t() * distance` as the `Q` matrix.

See the [examples section](#simple-examples) for more details.

***Caveat:*** NCA operates by repeatedly computing expressions of the form
`exp(-distance.Evaluate(data.col(i), data.col(j)))` (that is, the exponential of
the negative distance between two points).  When distances are very large, this
***quantity underflows to 0*** and results will not be reasonable.

 - This situation can be detected, usually by a result where `distance` is equal
   to the identity matrix.
 - Alternately, if the [`ens::ProgressBar()`
   callback](https://www.ensmallen.org/docs.html#progressbar) is used, a loss of
   0 often means this situation has occurred.
 - To mitigate the problem, consider scaling data such that the maximum pairwise
   distance is less than 10.  See the [simple examples](#simple-examples) that
   use the `vehicle` dataset.

#### `LearnDistance()` Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) training matrix. |
| `labels` | [`arma::Row<size_t>`](../matrices.md) | Training labels, [between `0` and `numClasses - 1`](../core/normalizing_labels.md) (inclusive).  Should have length `data.n_cols`.  |
| `distance` | [`arma::mat`](../matrices.md) | Output matrix to store transformation matrix representing learned distance. |
| `optimizer` | [any ensmallen optimizer](https://www.ensmallen.org) | Instantiated ensmallen optimizer for [differentiable functions](https://www.ensmallen.org/docs.html#differentiable-functions) or [differentiable separable functions](https://www.ensmallen.org/docs.html#differentiable-separable-functions). | `ens::StandardSGD()` |
| `callbacks...` | [any set of ensmallen callbacks](https://www.ensmallen.org/docs.html#callback-documentation) | Optional callbacks for the ensmallen optimizer, such as e.g. `ens::ProgressBar()`, `ens::Report()`, or others. | _(N/A)_ |

***Note***: any matrix type can be used for `data` and `distance`, so long as
that type implements the Armadillo API.  So, e.g., `arma::fmat` can be used.

### Other Functionality

 * An `NCA` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).
   Note that this is only meaningful if a custom `DistanceType` is being used,
   and that custom `DistanceType` has state to be saved.

 * `nca.Distance()` will return the `DistanceType` being used for learning.
   Unless a custom `DistanceType` was specified in the constructor,
   this simply returns a
   [`SquaredEuclideanDistance`](../core/distances.md#lmetric) object.

### Simple Examples

Learn a distance metric to improve classification performance on the iris
dataset, and show improved performance when using
[`NaiveBayesClassifier`](naive_bayes_classifier.md).

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("iris.labels.csv", labels, true);

// Create an NCA object and learn a distance.
arma::mat distance;
mlpack::NCA nca;
nca.LearnDistance(dataset, labels, distance);

// The distance matrix has size equal to the dimensionality of the data.
std::cout << "Learned distance size: " << distance.n_rows << " x "
    << distance.n_cols << "." << std::endl;

// Learn a NaiveBayesClassifier model on the data and print the performance.
mlpack::NaiveBayesClassifier nbc1(dataset, labels, 3);
arma::Row<size_t> predictions;
nbc1.Classify(dataset, predictions);
std::cout << "Naive Bayes Classifier without NCA: "
    << arma::accu(labels == predictions) << " of " << labels.n_elem
    << " correct." << std::endl;

// Now transform the data and learn another NaiveBayesClassifier.
arma::mat transformedDataset = distance * dataset;
mlpack::NaiveBayesClassifier nbc2(transformedDataset, labels, 3);
nbc2.Classify(transformedDataset, predictions);
std::cout << "Naive Bayes Classifier with NCA:    "
    << arma::accu(labels == predictions) << " of " << labels.n_elem
    << " correct." << std::endl;
```

---

Learn a distance metric on the ionosphere dataset, using 32-bit floating point
to represent the data and metric.

```c++
// See https://datasets.mlpack.org/ionosphere.csv.
arma::fmat dataset;
mlpack::data::Load("ionosphere.csv", dataset, true);

// The labels are the last row of the dataset.
arma::Row<size_t> labels =
    arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
dataset.shed_row(dataset.n_rows - 1);

// Create an NCA object and learn distance on float32 data.
// To keep computation time down, we use an instantiated optimizer that will
// only perform 10 epochs of training.  (In a real application you may want to
// train for longer!)
arma::fmat distance;
mlpack::NCA nca;

ens::StandardSGD opt;
opt.MaxIterations() = 10 * dataset.n_cols;
nca.LearnDistance(dataset, labels, distance, opt, ens::ProgressBar());

// We want to compute six quantities:
//
//  - Average distance to points of the same class before NCA.
//  - Average distance to points of the same class after NCA, using
//    MahalanobisDistance.
//  - Average distance to points of the same class after NCA, using the
//    transformed dataset.
//
//  - The same three quantities above, but for points of the other class.
//
// NCA should reduce the average distance to points in the same class, while
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
std::cout << " - Before NCA:                           " << distSums[0] << "."
    << std::endl;
std::cout << " - After NCA (with MahalanobisDistance): " << distSums[1] << "."
    << std::endl;
std::cout << " - After NCA (with transformed dataset): " << distSums[2] << "."
    << std::endl;
std::cout << std::endl;

std::cout << "Average distance between point 0 and points of other classes: "
    << std::endl;
std::cout << " - Before NCA:                           " << distSums[3] << "."
    << std::endl;
std::cout << " - After NCA (with MahalanobisDistance): " << distSums[4] << "."
    << std::endl;
std::cout << " - After NCA (with transformed dataset): " << distSums[5] << "."
    << std::endl;
std::cout << std::endl;

std::cout << "Ratio of other-class to same-class distances:" << std::endl;
std::cout << "(We expect this to go up.)" << std::endl;
std::cout << " - Before NCA: " << (distSums[3] / distSums[0]) << "."
    << std::endl;
std::cout << " - After NCA:  " << (distSums[5] / distSums[2]) << "."
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

arma::mat distance;
mlpack::NCA nca;

// Use a callback that prints a final optimization report.
nca.LearnDistance(dataset, labels, distance, lbfgs, ens::Report());
```

---

<!-- TODO: actually use a kNN classifier here... once we have it implemented! -->

Learn a distance metric on the vehicle dataset, but instead of using the
Euclidean distance as the underlying metric, use the Manhattan distance.  This
means that NCA is optimizing k-NN performance under the Manhattan distance, not
under the Euclidean distance.

```c++
// See https://datasets.mlpack.org/vehicle.csv.
arma::mat dataset;
mlpack::data::Load("vehicle.csv", dataset, true);

// The labels are contained as the last row of the dataset.
arma::Row<size_t> labels =
    arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
dataset.shed_row(dataset.n_rows - 1);

// Because typical distances between points in the vehicle dataset are large,
// we will center the dataset and scale it to have points in the unit ball.
// (That is, all points will have values in each dimension between -1 and 1.)
// This means that the maximum pairwise distance is 2.
dataset.each_col() -= arma::mean(dataset, 1);
dataset /= arma::max(arma::max(arma::abs(dataset)));

// Create the NCA object and optimize.  Use Nesterov momentum SGD, printing a
// progress bar during optimization.
mlpack::NCA<mlpack::ManhattanDistance> nca;
arma::mat distance;
ens::NesterovMomentumSGD opt(0.01 /* step size */,
                             32 /* batch size */,
                             20 * dataset.n_cols /* 20 epochs */);
nca.LearnDistance(dataset, labels, distance, opt, ens::ProgressBar());

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
std::cout << "   * Before NCA: " << d1 << std::endl;
std::cout << "   * After NCA:  " << d2 << std::endl;
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
std::cout << "   * Before NCA: " << d3 << std::endl;
std::cout << "   * After NCA:  " << d4 << std::endl;

// Note that point 3 has been moved further away from point 0 than point 1.
```

---

Learn a distance metric while also performing dimensionality reduction, reducing
the dimensionality of the vehicle dataset by 2 dimensions.

```c++
// See https://datasets.mlpack.org/vehicle.csv.
arma::mat dataset;
mlpack::data::Load("vehicle.csv", dataset, true);

// The labels are contained as the last row of the dataset.
arma::Row<size_t> labels =
    arma::conv_to<arma::Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
dataset.shed_row(dataset.n_rows - 1);

// Because typical distances between points in the vehicle dataset are large,
// we will center the dataset and scale it to have points in the unit ball.
// (That is, all points will have values in each dimension between -1 and 1.)
// This means that the maximum pairwise distance is 2.
dataset.each_col() -= arma::mean(dataset, 1);
dataset /= arma::max(arma::max(arma::abs(dataset)));

// Use a random initialization for the distance transformation, with the
// specified output dimensionality.
arma::mat distance(dataset.n_rows - 2, dataset.n_rows, arma::fill::randu);
mlpack::NCA nca;
ens::L_BFGS opt;
opt.MaxIterations() = 10; // You may want more in a real application.
nca.LearnDistance(dataset, labels, distance, opt);

// Now transform the dataset.
arma::mat transformedData = distance * dataset;

std::cout << std::endl << std::endl;
std::cout << "Original data has size " << dataset.n_rows << " x "
    << dataset.n_cols << "." << std::endl;
std::cout << "Transformed data has size " << transformedData.n_rows << " x "
    << transformedData.n_cols << "." << std::endl;
```
