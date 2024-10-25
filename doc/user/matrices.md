# Matrices in mlpack

mlpack uses Armadillo matrices for linear algebra support.  Armadillo is a fast
C++ matrix library which uses advanced template metaprogramming techniques to
provide the fastest possible linear algebra operations.

<center><p><img src="https://arma.sourceforge.net/img/armadillo_logo2.png" alt="Armadillo logo"></p></center>

Detailed documentation on Armadillo can be found on [the Armadillo
website](https://arma.sourceforge.net/docs.html).

Nonetheless, there are a few further caveats for mlpack Armadillo usage.

 * [An Armadillo primer](#an-armadillo-primer)
 * [Representing data in mlpack](#representing-data-in-mlpack)
 * [Loading data](#loading-data)
 * [Loading and using categorical data](#loading-and-using-categorical-data)
 * [Alternate matrix types](#alternate-matrix-types)
 * [Adapting from other toolkits (Eigen, etc.)](#adapting-from-other-toolkits-eigen-etc)

## An Armadillo primer

The Armadillo syntax is straightforward and is aimed at ease-of-use and
readability.  To give a flavor of what a linear algebra program using Armadillo
looks like, see the trivial (contrived) program below that performs some basic
matrix operations.

```c++
// Create a 10x15 matrix with random elements.
arma::mat m(10, 15, arma::fill::randu);

std::cout << "Size of m: " << m.n_rows << " x " << m.n_cols << "." << std::endl;

// Sum all elements in the matrix.
const double sumVal = arma::accu(m);
std::cout << "Sum of all elements: " << sumVal << "." << std::endl;

// Sum the elements in each column.
arma::rowvec sums = arma::sum(m, 0);
std::cout << "Sums in each column: " << sums;

// Add 1 to all elements.
m += 1;

// Subtract sums from each row.
m.each_row() -= sums;

// Print an individual element.
std::cout << "m(3, 4) is: " << m(3, 4) << "." << std::endl;
```

For more information on Armadillo, see the following resources:

 * [Armadillo documentation](https://arma.sourceforge.net/docs.html)
 * [Armadillo example
    program](https://arma.sourceforge.net/docs.html#example_prog)
 * [Armadillo/MATLAB syntax conversion
   table](https://arma.sourceforge.net/docs.html#syntax)

## Representing data in mlpack

Armadillo matrices, unlike numpy and some other toolkits, store data in a
***column-major*** format.  This means that each column is located in contiguous
memory; i.e., `x(0, 0)` is adjacent to `x(1, 0)` in memory.

This means that, for the vast majority of machine learning methods, it is faster
to store ***observations as columns*** and ***dimensions as rows***.  This is
counter to most standard machine learning texts!  It also has some implications
for linear algebra operations; for instance, computing the Gram matrix of a
matrix `X` is typically expressed as `X^T X`, but when using column-major
matrices, the expression must be `X X^T`.

In general, the following Armadillo types are commonly used inside mlpack:

 * `arma::mat`: datasets and general-purpose matrices
 * `arma::Row<size_t>`: integer response data, e.g., labels for classification
   datasets
 * `arma::rowvec`: floating-point response data, e.g., responses for regression
   datasets
 * `arma::vec`: general-purpose column vectors
 * `arma::sp_mat`, `arma::fmat`: alternate types for representing data; see
   [Alternate matrix types](#alternate-matrix-types)

## Loading data

mlpack provides two simple functions for loading and saving data matrices in a
column-major form:

 * `data::Load(filename, matrix, fatal=false, transpose=true, type=FileType::AutoDetect)` ([full documentation](load_save.md#numeric-data))
 * `data::Save(filename, matrix, fatal=false, transpose=true, type=FileType::AutoDetect)` ([full documentation](load_save.md#numeric-data))

As an example, consider the following CSV file:

```sh
$ cat data.csv
3,3,3,3,0
3,4,4,3,0
3,4,4,3,0
3,3,4,3,0
3,6,4,3,0
2,4,4,3,0
2,4,4,1,0
3,3,3,2,0
3,4,4,2,0
3,4,4,2,0
3,3,4,2,0
3,6,4,2,0
2,4,4,2,0
```

The following program will load the data, print information about it, and save a
modified dataset to disk.

```c++
// Load data from `data.csv` into `m`.  Throw an exception on failure (i.e. set
// `fatal` to `true`).
arma::mat m;
mlpack::data::Load("data.csv", m, true);

// Since mlpack uses column-major data,
//
//  - each column corresponds to a data point!
//  - each row corresponds to a dimension!
//
std::cout << "The matrix in 'data.csv' has: " << std::endl;
std::cout << " - " << m.n_cols << " points." << std::endl;
std::cout << " - " << m.n_rows << " dimensions." << std::endl;

std::cout << "The second point in the dataset: " << std::endl;
std::cout << m.col(1).t();

// Now modify the matrix and save to a different format (space-separated
// values).
m += 3;
mlpack::data::Save("data-mod.txt", m);
```

Although Armadillo does provide a `.load()` and `.save()` member function for
matrices, the `data::Load()` and `data::Save()` functions offer additional
flexibility, and ensure that data is saved and loaded in a column-major format.

## Loading and using categorical data

Some mlpack techniques support mixed categorical data, e.g., data where some
dimensions take only categorical values (e.g. `0`, `1`, `2`, etc.).  String data
and other non-numerical data can be represented as categorical values, and
mlpack has support to load mixed categorical data:

 * The `data::DatasetInfo` auxiliary class stores information about whether each
   dimension is numeric or categorical.  ([full
   documentation](load_save.md#datadatasetinfo))
 * `data::Load(filename, matrix, info, fatal=false, transpose=true)` ([full
   documentation](load_save.md#loading-categorical-data))

For example, consider the following CSV file that contains strings:

```sh
$ cat mixed_string_data.csv
3,"hello",3,"f",0
3,"goodbye",4,"f",0
3,"goodbye",4,"e",0
3,"hello",4,"d",0
3,"hello",4,"d",0
2,"hello",4,"d",0
2,"hello",4,"d",0
3,"goodbye",3,"f",0
3,"goodbye",4,"f",0
3,"hello",4,"f",0
3,"hello",4,"c",0
3,"hello",4,"f",0
2,"hello",4,"c",0
```

The following program will load the data file, print information about
categorical dimensions, and prepare the data for use with an mlpack algorithm
that supports mixed categorical data.

```c++
// Load data from `mixed_string_data.csv` into `m`.  Throw an exception on
// failure (i.e. set `fatal` to `true`).  This populates the `info` object.
arma::mat m;
mlpack::data::DatasetInfo info;
mlpack::data::Load("mixed_string_data.csv", m, info, true);

// Print information about the data.
std::cout << "The matrix in 'mixed_string_data.csv' has: " << std::endl;
std::cout << " - " << m.n_cols << " points." << std::endl;
std::cout << " - " << info.Dimensionality() << " dimensions." << std::endl;

// Print which dimensions are categorical.
for (size_t d = 0; d < info.Dimensionality(); ++d)
{
  if (info.Type(d) == mlpack::data::Datatype::categorical)
  {
    std::cout << " - Dimension " << d << " is categorical with "
        << info.NumMappings(d) << " distinct categories." << std::endl;
  }
}

// Modify the third point to be 4,"wonderful",1,"c",0.
// Note that we manually map the string values; MapString() returns the category
// for a given value.
m(0, 2) = 4;
m(1, 2) = info.MapString<double>("wonderful", 1); // Create new third category.
m(2, 2) = 1;
m(3, 2) = info.MapString<double>("c", 1);
m(4, 2) = 0;

// `m` can now be used with any mlpack algorithm that supports categorical data.
```

Not every mlpack method supports categorical data.  Below are the list of
methods that do have categorical data support:

 * [`DecisionTree`](methods/decision_tree.md)
 * [`DecisionTreeRegressor`](methods/decision_tree_regressor.md)
 * [`RandomForest`](methods/random_forest.md)
 * [`HoeffdingTree`](methods/hoeffding_tree.md)

## Alternate matrix types

mlpack's documentation focuses on the use of the `arma::mat`, `arma::vec`, and
`arma::rowvec` types, with an underlying `double` numeric type, e.g., 64-bit
floating-point.  But, many mlpack algorithms and support utilities have support
for alternate matrix types and element types:

 * Many methods, such as
   [`LogisticRegression`](methods/logistic_regression.md#advanced-functionality-different-element-types),
   allow specifying the matrix types as a template parameter.

 * Some methods, such as
   [`DecisionTree`](methods/decision_tree.md#using-different-element-types),
   accept different matrix types to `Train()`, `Classify()`, or `Predict()`,
   without needing to specify an explicit template parameter.

 * In general, any matrix type that supports the Armadillo API can be used; this
   includes:
    - Single-precision floating point matrices (`arma::fmat`, `arma::frowvec`,
      `arma::fvec`)
    - Sparse matrices (`arma::sp_mat`, `arma::sp_fmat`)
    - GPU matrices via [Bandicoot](https://coot.sourceforge.io) (`coot::mat`,
      `coot::fmat`) --- ***(note: support is under development and still
       experimental)***

---

A simple example of using single-precision floating point data to train an
[`AdaBoost`](methods/adaboost.md) model is below.

```c++
// 1000 random points in 10 dimensions, using 32-bit precision (float).
arma::fmat dataset(10, 1000, arma::fill::randu);
// Random labels for each point, totaling 5 classes.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 4));

// Train in the constructor, using floating-point data.
// The weak learner type is now a floating-point Perceptron.
using PerceptronType = mlpack::Perceptron<
    mlpack::SimpleWeightUpdate,
    mlpack::ZeroInitialization,
    arma::fmat>;
mlpack::AdaBoost<PerceptronType, arma::fmat> ab(dataset, labels, 5);

// Create test data (500 points).
arma::fmat testDataset(10, 500, arma::fill::randu);
arma::Row<size_t> predictions;
ab.Classify(testDataset, predictions);
// Now `predictions` holds predictions for the test dataset.

// Print some information about the test predictions.
std::cout << arma::accu(predictions == 3) << " test points classified as class "
    << "3." << std::endl;
```

---

A simple example of using sparse 32-bit floating point data to train a
[`LogisticRegression`](methods/logistic_regression.md) model is below.

```c++
// Create random, sparse 100-dimensional data.
arma::sp_fmat dataset;
dataset.sprandu(100, 5000, 0.3);
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(5000, arma::distr_param(0, 1));

// Train with L2 regularization penalty parameter of 0.1.
mlpack::LogisticRegression<arma::sp_fmat> lr(dataset, labels, 0.1);

// Now classify a test point.
arma::sp_fvec point;
point.sprandu(100, 1, 0.3);

size_t prediction;
arma::fvec probabilitiesVec;
lr.Classify(point, prediction, probabilitiesVec);

std::cout << "Prediction for random test point: " << prediction << "."
    << std::endl;
std::cout << "Class probabilities for random test point: "
    << probabilitiesVec.t();
```

<!-- TODO: a simple Bandicoot example! -->

## Adapting from other toolkits (Eigen, etc.)

In general, C++ linear algebra toolkits store data in a column-major
representation, and transitioning between toolkits is a matter of getting access
to the underlying memory.

---

Copy an [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) matrix
into an Armadillo matrix.

```c++
// Note: this will only work if the Eigen matrix is stored in column-major
// order.  See https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
// for more details.
Eigen::MatrixXd m;
const size_t rows = 10;
const size_t cols = 20;
m.setRandom(rows, cols); // 10x20 random matrix.

// Copy into an Armadillo matrix.
arma::mat mCopy(&m(0, 0), rows, cols);
```

---

Copy an [XTensor](https://xtensor.readthedocs.io/en/latest/) matrix into an
Armadillo matrix.

```c++
// Note: this will only work correctly if the layout_type of the XTensor matrix
// is column-major (i.e. xt::layout_type::column_major).  See
// https://xtensor.readthedocs.io/en/latest/container.html for more details.
const size_t rows = 10;
const size_t cols = 20;

// Create a random 10 x 20 matrix with normally distributed values.
// Note that we must ensure that the matrix is laid out in column-major form.
xt::xarray<double, xt::layout_type::column_major> m =
    xt::random::randn<double>({ rows, cols });

// Copy into an Armadillo matrix.
arma::mat mCopy(m.data(), rows, cols);
```

---

Make an Armadillo matrix that is an alias of an
[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) matrix.  Note
that changes to the Eigen matrix will be reflected in the Armadillo matrix, and
vice versa.

*If the Eigen matrix is deallocated, the Armadillo matrix will become invalid.
Be careful!  [More details
here](https://arma.sourceforge.net/docs.html#adv_constructors_mat).*

```c++
// Note: this will only work if the Eigen matrix is stored in column-major
// order.  See https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
// for more details.
Eigen::MatrixXd m;
const size_t rows = 10;
const size_t cols = 20;
m.setRandom(rows, cols); // 10x20 random matrix.

// Make an Armadillo matrix that is an alias of the Eigen matrix.  This avoids
// the copy, but is potentially dangerous: be careful that the Eigen matrix is
// not deleted while the Armadillo matrix is in use!
//
// See https://arma.sourceforge.net/docs.html#adv_constructors_mat
arma::mat mAlias(&m(0, 0), rows, cols, false, true);
```

---

Copy an Armadillo matrix to an Eigen matrix.

```c++
const size_t rows = 10;
const size_t cols = 20;
arma::mat m(10, 20, arma::fill::randu);

// Construct the Eigen matrix by using a map of the Armadillo memory.
Eigen::MatrixXd eigenM(Eigen::Map<Eigen::MatrixXd>(m.memptr(), rows, cols));
```
