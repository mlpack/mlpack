# Imputation

mlpack provides functionality for replacing missing values in a dataset either
with imputed values, user-specified values, or removing points from a dataset
that have missing values entirely.

Removing missing values is an important part of the data science pipeline, as
mlpack's machine learning techniques do not support learning on data that
contain missing values.

 * [`Imputer`](#imputer): the class that is used for imputation.

 * [Imputation strategies](#imputation-strategies): the set of imputation
   strategies supported by `Imputer`.
   - [`MeanImputation`](#meanimputation): replace missing values with the mean
     value in a row or column.
   - [`MedianImputation`](#medianimputation): replace missing values with the
     median value in a row or column.
   - [`ListwiseDeletion`](#listwisedeletion): remove any data points that
     contain missing values.
   - [`CustomImputation`](#customimputation): replace missing values with a
     user-specified custom value.
   - [Custom imputation strategies](#custom-imputation-strategies): use a fully
     custom strategy to impute missing values

 * [Simple examples](#simple-examples) of imputing missing values with
   `Imputer`.

## `Imputer`

The `Imputer` class offers a simple interface to impute missing values into a
single dimension of a data matrix.

### Constructor

 * `imp = Imputer()`
   - Construct an imputer using the default imputation strategy
     ([`MeanImputation`](#meanimputation)).

 * `imp = Imputer<Strategy>()`
 * `imp = Imputer<Strategy>(strategy)`
   - Construct an imputer, specifying the imputation strategy manually.
   - `Strategy` should be [`MeanImputation`](#meanimputation),
     [`MedianImputation`](#medianimputation),
     [`ListwiseDeletion`](#listwisedeletion),
     [`CustomImputation<>`](#customimputation), or a
     [custom imputation class](#custom-imputation-strategies).
   - Optionally, specify an instantiated imputation strategy (`strategy`); this
     is only useful with [`CustomImputation<>`](#customimputation) or a
     [custom imputation class](#custom-imputation-strategies).

### `Impute()`

Once an `Imputer` object is constructed, the `Impute()` function can be used to
replace missing values.

 * `imp.Impute(data, missingValue, dimension)`
 * `imp.Impute(data, missingValue, dimension, columnMajor=true)`
   - Replace instances of `missingValue` in row `dimension` (a `size_t`) of the
     given matrix `data`.
   - `data` should be a
     [column-major data matrix](../matrices.md#representing-data-in-mlpack);
     if `columnMajor` is set to `false`, then missing values in *column*
     `dimension` (instead of row `dimension`) will be replaced.
   - `missingValue` should be the same type as the element type of `data` (e.g.
     if `data` is `arma::mat`, then `missingValue` should be `double`).

***Notes***:

 * Any missing value can be chosen for `missingValue`, but using
   [NaN](https://cplusplus.com/reference/cmath/nan-function/) is a good choice.

 * If imputation in the entire data matrix is desired, using
   [`data.replace()`](https://arma.sourceforge.net/docs.html#replace) is likely
   an easier and faster approach.  `Imputer` does not support this because most
   imputation strategies depend on data specific to a single dimension, not the
   entire matrix.

<!-- TODO: when MissingToNan() support is fully documented, we can update the
first bullet point to mention that it works well in a pipeline where you load
with MissingIsNan() to get NaNs, then impute -->

## Imputation strategies

mlpack provides four imputation strategies that can be used with `Imputer`.  It
is also possible to write a
[fully custom imputation strategy](#custom-imputation-strategies).

### `MeanImputation`

 * `MeanImputation` computes the mean of non-missing elements in a dimension and
   uses this value to replace missing elements.

 * The constructor of `MeanImputation` has no parameters, and so passing an
   instantiated `strategy` to the [constructor of `Imputer`](#constructor) is
   not necessary.

 * `MeanImputation` does not support imputation on sparse matrix (e.g.
   `arma::sp_mat`); use dense matrices (e.g. `arma::mat`) instead.

 * For more details, see
   [the source code](/src/mlpack/core/data/imputation_strategies/mean_imputation.hpp).

### `MedianImputation`

 * `MedianImputation` computes the median of non-missing elements in a dimension
   and uses this value to replace missing elements.

 * The constructor of `MedianImputation` has no parameters, and so passing an
   instantiated `strategy` to the [constructor of `Imputer`](#constructor) is
   not necessary.

 * `MedianImputation` does not support imputation on sparse matrix (e.g.
   `arma::sp_mat`); use dense matrices (e.g. `arma::mat`) instead.

 * For more details, see
   [the source code](/src/mlpack/core/data/imputation_strategies/median_imputation.hpp).

### `ListwiseDeletion`

 * `ListwiseDeletion` removes any data points that contain missing elements in
   the specified dimension to [`Impute()`](#impute).

 * The constructor of `ListwiseDeletion` has no parameters, and so passing an
   instantiated `strategy` to the [constructor of `Imputer`](#constructor) is
   not necessary.

 * `ListwiseDeletion` does not support imputation on sparse matrix (e.g.
   `arma::sp_mat`); use dense matrices (e.g. `arma::mat`) instead.

 * For more details, see
   [the source code](/src/mlpack/core/data/imputation_strategies/listwise_deletion.hpp).

### `CustomImputation`

 * `CustomImputation` replaces any missing elements in a dimension with a
   specified value.

 * `c = CustomImputation<>(value)` creates a `CustomImputation` object for use
   with the [constructor of `Imputer`](#constructor), which will replace missing
   values with `value` (specified as a `double`) when [`Impute()`](#impute) is
   called.

 * `value` will be casted to the element type of the data matrix when
   [`Impute()`](#impute) is called.

 * If a different element type is used for the data matrix,
   `c = CustomImputation<T>(value)` can be used to specify a `value` of the
   desired type `T` (e.g. `float`, `int`, etc.).  In this situation, the class
   `Imputer<CustomImputation<T>>` will need to be used.

 * For more details, see
   [the source code](/src/mlpack/core/data/imputation_strategies/custom_imputation.hpp).

### Custom imputation strategies

Implementing a fully custom imputation strategy requires a class with only one
method, matching the following API:

```c++
class FullyCustomImputation
{
 public:
  // This function should replace values in `data` in row `dimension` that have
  // value `missingValue` with whatever the custom imputation strategy value
  // should be.
  //
  // If `columnMajor` is `false`, then values should be replaced in the column
  // `dimension` instead.
  //
  // Remember when checking for missing values that NaN != NaN---use
  // `std::isnan()` instead to check if `missingValue` is NaN, and to check if
  // values in `data` are NaN.
  //
  // `data` will be an Armadillo matrix type or equivalent type matching the
  // Armadillo API.
  template<typename MatType>
  void Impute(MatType& data,
              const typename MatType::elem_type missingValue,
              const size_t dimension,
              const bool columnMajor = true);
};
```

## Simple examples

Replace all NaNs in dimension 2 of a random matrix with the mean in that
dimension.

```c++
// Create a random matrix with integer values that are either 0, 1, or NaN.
arma::mat data = arma::randi<arma::mat>(10, 20, arma::distr_param(0, 2));
// Replace the value 2 with NaN.
data.replace(2, std::nan(""));

mlpack::Imputer<mlpack::MeanImputation> imputer;

std::cout << "Dimension 2 before imputation:" << std::endl;
std::cout << data.row(2);

imputer.Impute(data, std::nan(""), 2);

std::cout << "Dimension 2 after imputation:" << std::endl;
std::cout << data.row(2);
```

---

Replace the value 0.0 in dimension 3 of a random matrix with the median in that
dimension.

```c++
// Create a random matrix with NaNs and random values in [0.5, 1].
arma::mat data(10, 20, arma::fill::randu);
// Replace anything below 0.5 with NaN.
data.transform([](double val) { return (val <= 0.5) ? std::nan("") : val; });

mlpack::Imputer<mlpack::MedianImputation> imputer;

std::cout << "Dimension 3 before imputation:" << std::endl;
std::cout << data.row(3);

imputer.Impute(data, std::nan(""), 3);

std::cout << "Dimension 3 after imputation:" << std::endl;
std::cout << data.row(3);
```

---

Remove any columns where dimension 4 contains a NaN value from a random matrix.

```c++
// Create a random matrix with values in [0, 1].  In dimension 4, any value less
// than 0.3 will be turned into a NaN.
arma::mat data(10, 1000, arma::fill::randu);
data.row(4).transform(
    [](double val) { return (val < 0.3) ? std::nan("") : val; });

mlpack::Imputer<mlpack::ListwiseDeletion> imputer;

std::cout << "Dataset contains " << data.n_cols << " points before removing "
    << "points that have NaN in dimension 4." << std::endl;

imputer.Impute(data, std::nan(""), 4);

std::cout << "Dataset contains " << data.n_cols << " points after removing "
    << "points that have NaN in dimension 4." << std::endl;
```

---

Replace the value 0.0 in dimension 0 of a random matrix with the value 2.5.  Use
a 32-bit floating point matrix as the data type.

```c++
// Create random matrix with values in [0, 5].
arma::fmat data = arma::randi<arma::fmat>(5, 20, arma::distr_param(0, 5));

mlpack::CustomImputation<> c(2.5); // Replace values with 2.5.
mlpack::Imputer<mlpack::CustomImputation<>> imputer(c);

std::cout << "Dimension 0 before imputation:" << std::endl;
std::cout << data.row(0);

imputer.Impute(data, 0.0, 0);

std::cout << "Dimension 0 after imputation:" << std::endl;
std::cout << data.row(0);
```
