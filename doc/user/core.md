# mlpack core class documentation

Underlying the implementations of [mlpack's machine learning
algorithms](../index.md#mlpack-algorithm-documentation) are mlpack core support
classes, each of which are documented on this page.

 * [Core math utilities](#core-math-utilities): utility classes for mathematical
   purposes
 * [Distances](#distances): distance metrics for geometric algorithms
 * [Distributions](#distributions): probability distributions
 * [Kernels](#kernels): Mercer kernels for kernel-based algorithms

## Core math utilities

mlpack provides a number of additional mathematical utility classes and
functions on top of Armadillo.

 * [Aliases](#aliases): utilities to create and manage aliases (`MakeAlias()`,
   `ClearAlias()`, `UnwrapAlias()`).

 * [`Range`](#range): simple mathematical range (i.e. `[0, 3]`)

 * [`ColumnCovariance()`](#columncovariance): compute covariance of
   [column-major](matrices.md#representing-data-in-mlpack) data

 * [`ColumnsToBlocks`](#columnstoblocks): reshape data points into a block
   matrix for visualization (useful for images)

 * [Distribution utilities](#distribution-utilities): `Digamma()`, `Trigamma()`

 * [`RandVector()`](#randvector): generate random vector on the unit sphere
   using the Box-Muller transform

 * [Logarithmic utilities](#logarithmic-utilities): `LogAdd()`, `AccuLog()`,
   `LogSumExp()`, `LogSumExpT()`.

 * [`MultiplyCube2Cube()`](#multiplycube2cube): multiply each slice in a cube by each slice in another cube
 * [`MultiplyMat2Cube()`](#multiplymat2cube): multiply a matrix by each slice in a cube
 * [`MultiplyCube2Mat()`](#multiplycube2mat): multiply each slice in a cube by a matrix
 * [`Quantile()`](#quantile): compute the quantile function of the Gaussian
   distribution

 * [RNG and random number utilities](#rng-and-random-number-utilities): extended
   scalar random number generation functions
 * [`RandomBasis()`](#randombasis): generate a random orthogonal basis
 * [`ShuffleData()`](#shuffledata): shuffle a dataset and associated labels

---

### Aliases

Aliases are matrix, vector, or cube objects that share memory with another
matrix, vector, or cube.  They are often used internally inside of mlpack to
avoid copies.

***Important caveats about aliases***:

 - An alias represents the same memory block as the input.  As such, changes to
   the alias object will also be reflected in the original object.

 - The `MakeAlias()` function is not guaranteed to return an alias; it only
   returns an alias *if possible*, and makes a copy otherwise.

 - If `mat` goes out of scope or is destructed, then `a` ***becomes invalid***.
   _You are responsible for ensuring an invalid alias is not used!_

---

 * `MakeAlias(a, vector, rows, cols, offset=0, strict=true)`
   - Make `a` into an alias of `vector` with the given size.
   - If `offset` is `0`, then the alias is identical: the first element of
     `a` is the first element of `vector`. Otherwise, the first element of `a`
     is the `offset`'th element of `vector`.
   - If `strict` is `true`, the size of `a` cannot be changed.
   - `vector` and `a` should have the same vector type (e.g. `arma::vec`,
     `arma::fvec`).
   - If an alias cannot be created, the vector will be copied.

 * `MakeAlias(a, mat, rows, cols, offset=0, strict=true)`
   - Make `a` into an alias of `mat` with the given size.
   - If `offset` is `0`, then the alias is identical: the first element of
     `a` is the first element of `mat`. Otherwise, the first element of `a`
     is the `offset`'th element of `mat`; elements in `mat` are ordered in
     a [column-major way](matrices.md#representing-data-in-mlpack).
   - If `strict` is `true`, the size of `a` cannot be changed.
   - `mat` and `a` should have the same matrix type (e.g. `arma::mat`,
     `arma::fmat`, `arma::sp_mat`).
   - If an alias cannot be created, the matrix will be copied.  Sparse types
     cannot have aliases and will be copied.

 * `MakeAlias(a, cube, rows, cols, slices, offset=0, strict=true)`
   - Make `a` into an alias of `cube` with the given size.
   - If `offset` is `0`, then the alias is identical: the first element of
     `a` is the first element of `cube`. Otherwise, the first element of `a`
     is the `offset`'th element of `cube`; elements in `cube` are ordered in
     a [column-major way](matrices.md#representing-data-in-mlpack).
   - If `strict` is `true`, the size of `a` cannot be changed.
   - `cube` and `a` should have the same cube type (e.g. `arma::cube`,
     `arma::fcube`).
   - If an alias cannot be created, the cube will be copied.

---

 * `ClearAlias(a)`
   - If `a` is an alias, reset `a` to an empty matrix, without modifying the
     aliased memory.  `a` is no longer an alias after this call.

---

 * `UnwrapAlias(a, in)`
   - If `in` is a matrix type (e.g. `arma::mat`), make `a` into an alias of
     `in`.
   - If `in` is not a matrix type, but instead, e.g., an Armadillo expression,
     fill `a` with the results of the evaluated expression `in`.
   - This can be used in place of, e.g., `a = in`, to avoid a copy when
     possible.
   - `a` should be a matrix type that matches the type of the expression or
     matrix `in`.

---

### `Range`

The `Range` class represents a simple mathematical range (i.e. `[0, 3]`),
with the bounds represented as `double`s.

---

#### Constructors

 * `r = Range()`
   - Construct an empty range.

 * `r = Range(p)`
   - Construct the range `[p, p]`.

 * `r = Range(lo, hi)`
   - Construct the range `[lo, hi]`.

---

#### Accessing and modifying range properties

 * `r.Lo()` and `r.Hi()` return the lower and upper bounds of the range as
   `double`s.
   - A range is considered empty if `r.Lo() > r.Hi()`.
   - These can be used to modify the bounds, e.g., `r.Lo() = 3.0`.

 * `r.Width()` returns the span of the range (i.e. `r.Hi() - r.Lo()`) as a
   `double`.

 * `r.Mid()` returns the midpoint of the range as a `double`.

---

#### Working with ranges

 * Given two ranges `r1` and `r2`,
   - `r1 | r2` returns the union of the ranges,
   - `r1 |= r2` expands `r1` to include the range `r2`,
   - `r1 & r2` returns the intersection of the ranges (possibly an empty range),
   - `r1 &= r2` shrinks `r1` to the intersection of `r1` and `r2`,
   - `r1 == r2` returns `true` if the two ranges are strictly equal (i.e. lower
     and upper bounds are equal),
   - `r1 != r2` returns `true` if the two ranges are not strictly equal,
   - `r1 < r2` returns `true` if `r1.Hi() < r2.Lo()`,
   - `r1 > r2` returns `true` if `r1.Lo() > r2.Hi()`, and
   - `r1.Contains(r2)` returns `true` if the ranges overlap at all.

 * Given a range `r` and a `double` scalar `d`,
   - `r * d` returns a new range `[d * r.Lo(), d * r.Hi()]`,
   - `r *= d` scales `r.Lo()` and `r.Hi()` by `d`, and
   - `r.Contains(d)` returns `true` if `d` is contained in the range.

---

 * To use ranges with different element types (e.g. `float`), use the type
   `RangeType<float>` or similar.

---

Example:

```c++
mlpack::Range r1(5.0, 6.0); // [5, 6]
mlpack::Range r2(7.0, 8.0); // [7, 8]

mlpack::Range r3 = r1 | r2; // [5, 8]
mlpack::Range r4 = r1 & r2; // empty range

bool b1 = r1.Contains(r2); // false
bool b2 = r1.Contains(5.5); // true
bool b3 = r1.Contains(r3); // true
bool b4 = r3.Contains(r4); // false

// Create a range of `float`s and a range of `int`s.
mlpack::RangeType<float> r5(1.0f, 1.5f); // [1.0, 1.5]
mlpack::RangeType<int> r6(3, 4); // [3, 4]
```

---

`Range` is used by:

 * [`RangeSearch`](/src/mlpack/methods/range_search/range_search.hpp)
 * [mlpack trees](../developer/trees.md) <!-- TODO: link to local trees section -->

---

### `ColumnCovariance()`

 * `ColumnCovariance(X, normType=0)`
   - `X`: a [column-major](matrices.md#representing-data-in-mlpack) data matrix
   - `normType`: either `0` or `1` (see below)

 * Computes the covariance of the data matrix `X`.

 * Equivalent to `arma::cov(X.t(), normType)`, but avoids computing the
   transpose and is thus slightly more efficient.

 * `normType` controls the type of normalization done when computing the
   covariance:
   - `0` will normalize with `X.n_cols - 1`, providing the best unbiased
     estimation of the covariance matrix (if the columns are from a normal
     distribution);
   - `1` will normalize with `X.n_cols`, providing the second moment about the
     mean of the columns.

 * Any dense matrix type can be used so long as it supports the Armadillo API
   (e.g., `arma::mat`, `arma::fmat`, etc.).

Example:

```c++
// Generate a random data matrix with 100 points in 5 dimensions.
arma::mat data(5, 100, arma::fill::randu);

// Compute the covariance matrix of the column-major matrix.
arma::mat cov = mlpack::ColumnCovariance(data);
cov.print("Covariance of random matrix:");
```

---

### `ColumnsToBlocks`

The `ColumnsToBlocks` class provides a way to transform data points (e.g.
columns in a matrix) into a block matrix format, primarily useful for
visualization as an image.

As a simple example, given a matrix with four columns `A`, `B`, `C`, and `D`,
`ColumnsToBlocks` can transform this matrix into the form

```
[[m m m m m]
 [m A m B m]
 [m m m m m]
 [m C m D m]
 [m m m m m]]
```

where `m` is a margin, and where each column may itself be reshaped into a
block.

---

#### Constructors

 * `ctb = ColumnsToBlocks(rows, cols)`
   - Create a `ColumnsToBlocks` object that will reshape the input matrix into
     blocks of shape `rows` by `cols`.
   - Each input column will be reshaped into a square (e.g. `ctb.BlockHeight()`
     and `ctb.BlockWidth()` are set to `0`).

 * `ctb = ColumnsToBlocks(rows, cols, blockHeight, blockWidth)`
   - Create a `ColumnsToBlocks` object that will reshape the input matrix into
     blocks of shape `rows` by `cols`.
   - Each individual column will also be reshaped into a block of shape
     `blockHeight` by `blockWidth`.

---

#### Properties

 * `ctb.Rows(rows)` will set the number of rows in the block output to `rows`.
   - `ctb.Rows()` will return a `size_t` with the current setting.

 * `ctb.Cols(cols)` will set the number of columns in the block output to
   `cols`.
   - `ctb.Cols()` will return a `size_t` with the current setting.

 * `ctb.BlockHeight(blockHeight)` will set the number of rows in each individual
   block to `blockHeight`.
   - `ctb.BlockHeight()` will return a `size_t` with the current setting.
   - If `ctb.BlockHeight()` is `0`, each input column will be reshaped into a
     square; if this is not possible, an exception will be thrown.

 * `ctb.BlockWidth()` will set the number of columns in each individual block to
   `blockWidth`.
   - `ctb.BlockWidth()` will return a `size_t` with the current setting.
   - If `ctb.BlockWidth()` is `0`, each input column will be reshaped into a
     square; if this is not possible, an exception will be thrown.

 * `ctb.BufSize(bufSize)` will set the number of margin elements to `bufSize`.
   - `ctb.BufSize()` will return a `size_t` with the current setting.
   - The default setting is `1`.

 * `ctb.BufValue(bufValue)` will set the element used for margins to `bufValue`.
   - `ctb.BufValue()` will return a `size_t` with the current setting.
   - The default setting is `-1.0`.

---

### Scaling values

`ColumnsToBlocks` also has the capability of linearly scaling values of the
inputs to a given range.

 * `ctb.Scale(true)` enables scaling values.
   - By default scaling is disabled.
   - `ctb.Scale(false)` will disable scaling.
   - `ctb.Scale()` will return a `bool` indicating whether scaling is enabled.

 * `ctb.MinRange(value)` sets the lower bound of the scaling range to `value`.
   - `ctb.MinRange()` returns the current value as a `double`.

 * `ctb.MaxRange(value)` sets the upper bound of the scaling range to `value`.
   - `ctb.MaxRange()` returns the current value as a `double`.
   - Must be greater than `ctb.MinRange()`, if `ctb.Scale() == true`.

***Note:*** the margin element (`ctb.BufValue()`) is considered during the
scaling process.

---

#### Transforming into block format

 * `ctb.Transform(input, output)` will perform the columns-to-blocks
   transformation on the given matrix `input`, storing the result in the matrix
   `output`.
   - An exception will be thrown if `input.n_rows` is not equal to
     `ctb.BlockHeight() * ctb.BlockWidth()` (if neither of those are `0`).
   - If either `ctb.BlockHeight()` or `ctb.BlockWidth()` is `0`, each column
     will be reshaped into a square, and an exception will be thrown if
     `input.n_rows` is not a perfect square (i.e. if `sqrt(input.n_rows)` is not
     an integer).

---

#### Examples

Reshape two 4-element vectors into one row of two blocks.

```c++
// This matrix has two columns.
arma::mat input;
input = { { -1.0000, 0.1429 },
          { -0.7143, 0.4286 },
          { -0.4286, 0.7143 },
          { -0.1429, 1.0000 } };
input.print("Input columns:");

arma::mat output;
mlpack::ColumnsToBlocks ctb(1, 2);
ctb.Transform(input, output);

// The columns of the input will be reshaped as a square which is
// surrounded by padding value -1 (this value could be changed with the
// BufValue() method):
// -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000
// -1.0000  -1.0000  -0.4286  -1.0000   0.1429   0.7143  -1.0000
// -1.0000  -0.7143  -0.1429  -1.0000   0.4286   1.0000  -1.0000
// -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000  -1.0000
output.print("Output using 2x2 block size:");

// Now, let's change some parameters; let's have each input column output not
// as a square, but as a 4x1 vector.
ctb.BlockWidth(1);
ctb.BlockHeight(4);
ctb.Transform(input, output);

// The output here will be similar, but each maximal input is 4x1:
// -1.0000 -1.0000 -1.0000 -1.0000 -1.0000
// -1.0000 -1.0000 -1.0000  0.1429 -1.0000
// -1.0000 -0.7143 -1.0000  0.4286 -1.0000
// -1.0000 -0.4286 -1.0000  0.7143 -1.0000
// -1.0000 -0.1429 -1.0000  1.0000 -1.0000
// -1.0000 -1.0000 -1.0000 -1.0000 -1.0000
output.print("Output using 4x1 block size:");
```

---

Load simple images and reshape into blocks.

```c++
// Load some favicons from websites associated with mlpack.
std::vector<std::string> images;
// See the following files:
// - https://datasets.mlpack.org/images/mlpack-favicon.png
// - https://datasets.mlpack.org/images/ensmallen-favicon.png
// - https://datasets.mlpack.org/images/armadillo-favicon.png 
// - https://datasets.mlpack.org/images/bandicoot-favicon.png
images.push_back("mlpack-favicon.png");
images.push_back("ensmallen-favicon.png");
images.push_back("armadillo-favicon.png");
images.push_back("bandicoot-favicon.png");

mlpack::data::ImageInfo info;
info.Channels() = 1; // Force loading in grayscale.

arma::mat matrix;
mlpack::data::Load(images, matrix, info, true);

// Now `matrix` has 4 columns, each of which is an individual image.
// Let's save that as its own image just for visualization.
mlpack::data::ImageInfo outInfo(matrix.n_cols, matrix.n_rows, 1);
mlpack::data::Save("favicons-matrix.png", matrix, outInfo, true);

// Use ColumnsToBlocks to create a 2x2 block matrix holding each image.
mlpack::ColumnsToBlocks ctb(2, 2);
ctb.BufValue(0.0); // Use 0 for the margin value.
ctb.BufSize(2); // Use 2-pixel margins.

arma::mat blocks;
ctb.Transform(matrix, blocks);

mlpack::data::ImageInfo blockOutInfo(blocks.n_cols, blocks.n_rows, 1);
mlpack::data::Save("favicons-blocks.png", blocks, blockOutInfo, true);
```

The resulting images (before and after using `ColumnsToBlocks`) are shown below.

*Before*:

<center>
<img src="../img/favicons-matrix.png" alt="four favicons each as a column in a matrix, unintelligible">
</center>

*After*:

<center>
<img src="../img/favicons-blocks.png" alt="four favicons each as a block in a larger image, much better">
</center>

---

#### See also

 * [Loading and saving image data](load_save.md#image-data)
 * [`SparseAutoencoder`](/src/mlpack/methods/sparse_autoencoder/sparse_autoencoder.hpp)

---

### Distribution utilities

 * `Digamma(x)` returns the logarithmic derivative of the gamma function (see
   [Wikipedia](https://en.wikipedia.org/wiki/Digamma_function)).
    - `x` should have type `double`.
    - The return type is `double`.

 * `Trigamma(x)` returns the [trigamma function](https://en.wikipedia.org/wiki/Trigamma_function) at the value `x`.
    - `x` should have type `double`.
    - The return type is `double`.

 * Both of these functions are used internally by the
   `GammaDistribution` class. <!-- TODO: document the class! -->

*Example*:

```
const double d1 = mlpack::Digamma(0.25);
const double d2 = mlpack::Digamma(1.0);

const double t1 = mlpack::Trigamma(0.25);
const double t2 = mlpack::Trigamma(1.0);

std::cout << "Digamma(0.25):  " << d1 << "." << std::endl;
std::cout << "Digamma(1.0):   " << d2 << "." << std::endl;
std::cout << "Trigamma(0.25): " << t1 << "." << std::endl;
std::cout << "Trigamma(1.0):  " << t2 << "." << std::endl;
```

---

### `RandVector()`

 * `RandVector(v)` generates a random vector on the unit sphere (i.e. with an
   L2-norm of 1) and stores it in `v` (an `arma::vec`).

 * The [Box-Muller transform](https://en.wikipedia.org/wiki/Box-Muller_transform)
   is used to generate the vector.

 * `v` is not resized, and should have size equal to the desired dimensionality
   when `RandVector()` is called.

*Example*:

```
// Generate a random 10-dimensional vector.
arma::vec v;
v.set_size(10);
RandVector(v);
v.print("Random 10-dimensional vector: ");

std::cout << "Random 10-dimensional vector: " << std::endl;
std::cout << v.t();
std::cout << "L2-norm of vector (should be 1): " << arma::norm(v, 2) << "."
    << std::endl;
```

### Logarithmic utilities

mlpack contains a few functions that are useful for working with logarithms, or
vectors containing logarithms.

 * `LogAdd(x, y)` for scalars `x` and `y` (e.g. `double`, `float`, `int`, etc.)
   will return `log(e^x + e^y)`.

 * `AccuLog(v)`, given a vector `v` containing log values, will return the
   scalar log-sum of those values:
   `log(e^(v[0]) + e^(v[1]) + ... + e^(v[v.n_elem - 1]))`.

---

 * `LogSumExp(m, out)`, given a matrix `m` (`arma::mat`) containing log values,
   will compute the scalar log-sum of each *column*, storing the result in the
   column vector `out` (type `arma::vec`).
   - `out` will be set to size `m.n_cols`.
   - `out[i]` will be equal to `AccuLog(m.col(i))`.
   - Different element types can be used for `m` and `out` (e.g. `arma::fmat`
     and `arma::fvec`).

 * `LogSumExpT(m, out)`, given a matrix `m` (type `arma::mat`) containing log
   values, will compute the scalar log-sum of each *row*, storing the result in
   the column vector `out` (type `arma::vec`)
   - `out` will be set to size `m.n_rows`.
   - `out[i]` will be equal to `AccuLog(m.row(i))`.
   - Different element types can be used for `m` and `out` (e.g. `arma::fmat`
     and `arma::fvec`).

---

 * `LogSumExp<eT, true>(m, out)` performs an incremental sum, otherwise
   identical to `LogSumExp()`.
   - The input values of `out` are not ignored.
   - `out[i]` will be equal to `log(e^(out[i]) + e^(AccuLog(m.col(i))))`.
   - `eT` represents the element type of `m` and `out` (e.g., `double` if `m` is
     `arma::mat` and `out` is `arma::vec`).

 * `LogSumExpT<eT, true>(m, out)` performs an incremental sum, otherwise
   identical to `LogSumExpT()`.
   - The input values of `out` are not ignored.
   - `out[i]` will be equal to `log(e^(out[i]) + e^(AccuLog(m.row(i))))`.
   - `eT` represents the element type of `m` and `out` (e.g., `double` if `m` is
     `arma::mat` and `out` is `arma::vec`).

---

### `MultiplyCube2Cube()`

 * `z = MultiplyCube2Cube(x, y, transX=false, transY=false)`
   - Inputs `x` and `y` are cubes (e.g. `arma::cube`), and must have the same
     number of slices
   - `z` is a cube whose slices are the slices of `x` and `y` multiplied
   - `transX` and `transY` indicate whether each slice of `x` and `y` should be
     transposed before multiplication.

 * If `transX` and `transY` are `false`, then
   `z.slice(i) = x.slice(i) * y.slice(i)`.

 * If `transX` is `false` and `transY` is `true`, then
   `z.slice(i) = x.slice(i) * y.slice(i).t()`.

 * The inner dimensions of `x` and `y` must match for multiplication, or an
   exception will be thrown.

*Example usage:*

```c++
// Generate two random cubes.
arma::cube x(10, 100, 5, arma::fill::randu); // 5 matrices, each 10x100.
arma::cube y(12, 100, 5, arma::fill::randu); // 5 matrices, each 12x100.

arma::cube z = mlpack::MultiplyCube2Cube(x, y, false, true);

// Output size should be 10x12x5.
std::cout << "Output size: " << z.n_rows << "x" << z.n_cols << "x" << z.n_slices
    << "." << std::endl;
```

---

### `MultiplyMat2Cube()`

 * `z = MultiplyMat2Cube(x, y, transX=false, transY=false)`
   - Input `x` is a matrix and `y` is a cube (e.g. `arma::cube`).
   - `z` is a cube whose slices are `x` multiplied by the slices of `y`.
   - `transX` and `transY` indicate whether `x` and each slice of `y` should be
     transposed before multiplication.

 * If `transX` and `transY` are `false`, then `z.slice(i) = x * y.slice(i)`.

 * If `transX` is `false` and `transY` is `true`, then
   `z.slice(i) = x * y.slice(i).t()`.

 * The inner dimensions of `x` and `y` must match for multiplication, or an
   exception will be thrown.

*Example usage:*

```c++
// Generate random inputs.
arma::mat  x(10, 100,    arma::fill::randu); // Random 10x100 matrix.
arma::cube y(12, 100, 5, arma::fill::randu); // 5 matrices, each 12x100.

arma::cube z = mlpack::MultiplyMat2Cube(x, y, false, true);

// Output size should be 10x12x5.
std::cout << "Output size: " << z.n_rows << "x" << z.n_cols << "x" << z.n_slices
    << "." << std::endl;
```

---

### `MultiplyCube2Mat()`

 * `z = MultiplyCube2Mat(x, y, transX=false, transY=false)`
   - Input `x` is a cube (e.g. `arma::cube`) and `y` is a matrix.
   - `z` is a cube whose slices are the slices of `x` multiplied with `y`.
   - `transX` and `transY` indicate whether each slice of `x` and `y` should be
     transposed before multiplication.

 * If `transX` and `transY` are `false`, then `z.slice(i) = x.slice(i) * y`.

 * If `transX` is `true` and `transY` is `false`, then
   `z.slice(i) = x.slice(i).t() * y`.

 * The inner dimensions of `x` and `y` must match for multiplication, or an
   exception will be thrown.

*Example usage:*

```c++
// Generate two random cubes.
arma::cube x(12, 50, 5, arma::fill::randu); // 5 matrices, each 12x50.
arma::mat  y(12, 60,    arma::fill::randu); // Random 12x60 matrix.

arma::cube z = mlpack::MultiplyCube2Mat(x, y, true, false);

// Output size should be 50x60x5.
std::cout << "Output size: " << z.n_rows << "x" << z.n_cols << "x" << z.n_slices
    << "." << std::endl;
```

---

### `Quantile()`

 * Compute the quantile function of the Gaussian distribution at the given
   probability.

 * `double q = Quantile(p, mu=0.0, sigma=1.0)`
   - `q` is the computed quantile.
   - `p` is the probability to compute the quantile of (between 0 and 1).
   - `mu` is the (optional) mean of the Gaussian distribution.
   - `sigma` is the (optional) standard deviation of the Gaussian distribution.
   - All arguments are `double`s.

 * See also [Quantile function on Wikipedia](https://en.wikipedia.org/wiki/Quantile_function).

*Example usage:*

```c++
// 70% of points from N(0, 1) are less than q1 = 0.524.
double q1 = mlpack::Quantile(0.7);

// 90% of points from N(0, 1) are less than q2 = 1.282.
double q2 = mlpack::Quantile(0.9);

// 50% of points from N(1, 1) are less than q3 = 1.0.
double q3 = mlpack::Quantile(0.5, 1.0); // Quantile of 1.0 for N(1, 1) is 1.0.

// 10% of points from N(1, 0.1) are less than q4 = 0.871.
double q4 = mlpack::Quantile(0.1, 1.0, 0.1);

std::cout << "Quantile(0.7): " << q1 << "." << std::endl;
std::cout << "Quantile(0.9): " << q2 << "." << std::endl;
std::cout << "Quantile(0.5, 1.0): " << q3 << "." << std::endl;
std::cout << "Quantile(0.1, 1.0, 0.1): " << q4 << "." << std::endl;
```

### RNG and random number utilities

On top of the random number generation support that Armadillo provides via
[randu()](https://arma.sourceforge.net/docs.html#randu),
[randn()](https://arma.sourceforge.net/docs.html#randn), and
[randi()](https://arma.sourceforge.net/docs.html#randi), mlpack provides
a few additional thread-safe random number generation functions for generating
random scalar values.

 * `RandomSeed(seed)` will set the random seed of mlpack's RNGs ***and***
   Armadillo's RNG to `seed`.
   - This internally calls `arma::arma_rng::set_seed()`.
   - In a multithreaded application, each thread's RNG will be deterministically
     set to a different value based on `seed`.

 * `Random()` returns a random `double` uniformly distributed between `0` and
   `1`, *not including 1*.

 * `Random(lo, hi)` returns a random `double` uniformly distributed between `lo`
   and `hi`, *not including `hi`*.

 * `RandBernoulli(p)` samples from a Bernoulli distribution with parameter `p`:
   with probability `p`, `1` is returned; with probability `1 - p`, `0` is
   returned.

 * `RandInt(hiExclusive)` returns a random `int` uniformly distributed in the
   range `[0, hiExclusive)`.

 * `RandInt(lo, hiExclusive)` returns a random `int` uniformly distributed in
   the range `[lo, hiExclusive)`.

 * `RandNormal()` returns a random `double` normally distributed with mean `0`
   and standard deviation `1`.

 * `RandNormal(mean, stddev)` returns a random `double` normally distributed
   with mean `mean` and standard deviation `stddev`.

*Examples*:

```c++
mlpack::RandomSeed(123); // Set a specific random seed.

const double r1 = mlpack::Random();             // In the range [0, 1).
const double r2 = mlpack::Random(3, 4);         // In the range [3, 4).
const double r3 = mlpack::RandBernoulli(0.25);  // P(1) = 0.25.
const int    r4 = mlpack::RandInt(10);          // In the range [0, 10).
const int    r5 = mlpack::RandInt(5, 10);       // In the range [5, 10).
const double r6 = mlpack::RandNormal();         // r6 ~ N(0, 1).
const double r7 = mlpack::RandNormal(2.0, 3.0); // r7 ~ N(2, 3).

std::cout << "Random():            " << r1 << "." << std::endl;
std::cout << "Random(3, 4):        " << r2 << "." << std::endl;
std::cout << "RandBernoulli(0.25): " << r3 << "." << std::endl;
std::cout << "RandInt(10):         " << r4 << "." << std::endl;
std::cout << "RandInt(5, 10):      " << r5 << "." << std::endl;
std::cout << "RandNormal():        " << r6 << "." << std::endl;
std::cout << "RandNormal(2, 3):    " << r7 << "." << std::endl;
```

---

### `RandomBasis()`

The `RandomBasis()` function generates a random d-dimensional orthogonal basis.

 * `RandomBasis(basis, d)` fills the matrix `basis` with `d` orthogonal vectors,
   each of dimension `d`.
   - `basis.col(i)` represents the `i`th basis vector.
   - `basis` will have size `d` rows by `d` cols.

 * The random basis is generated using the QR decomposition.

*Example*:

```c++
arma::mat basis;

// Generate a 10-dimensional random basis.
mlpack::RandomBasis(basis, 10);

// Each two vectors are orthogonal.
std::cout << "Dot product of basis vectors 2 and 4: "
    << arma::dot(basis.col(2), basis.col(4))
    << " (should be zero or very close!)." << std::endl;
```

---

### `ShuffleData()`

Shuffle a [column-major](matrices.md#representing-data-in-mlpack) dataset and
associated labels/responses, optionally with weights.  This preserves the
connection of each data point to its label (and optionally its weight).

 * `ShuffleData(inputData, inputLabels, outputData, outputLabels)`
   - Randomly permute data points and labels from `inputData` and `inputLabels`
     into `outputData` and `outputLabels`.
   - `outputData` will be set to the same size as `inputData`.
   - `outputLabels` will be set to the same size as `inputLabels`.
   - `inputData` can be a dense matrix, a sparse matrix, or a cube, with any
     element type.  (That is, `inputData` may have type `arma::mat`,
     `arma::fmat`, `arma::sp_mat`, `arma::cube`, etc.)
   - `inputLabels` must be a dense vector type but may hold any element type
     (e.g.  `arma::Row<size_t>`, `arma::uvec`, `arma::vec`, etc.).
   - `outputData` must have the same type as `inputData`, and `outputLabels`
     must have the same type as `inputLabels`.

 * `ShuffleData(inputData, inputLabels, inputWeights, outputData, outputLabels, outputWeights)`
   - Identical to the previous overload, but also handles weights via
     `inputWeights` and `outputWeights`.
   - `inputWeights` must be a dense vector type but may hold any element type
     (e.g.  `arma::rowvec`, `arma::frowvec`, `arma::vec`, etc.)
   - `outputWeights` must have the same type as `inputWeights`.

***Note:*** when `inputData` is a cube (e.g. `arma::cube` or similar), the
columns of the cube will be shuffled.

*Example usage:*

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset, true);
// See https://datasets.mlpack.org/iris.labels.csv.
arma::Row<size_t> labels;
mlpack::data::Load("iris.labels.csv", labels, true);

// Now shuffle the points in the iris dataset.
arma::mat shuffledDataset;
arma::Row<size_t> shuffledLabels;
mlpack::ShuffleData(dataset, labels, shuffledDataset, shuffledLabels);

std::cout << "Before shuffling, the first point was: " << std::endl;
std::cout << "  " << dataset.col(0).t();
std::cout << "with label " << labels[0] << "." << std::endl;
std::cout << std::endl;
std::cout << "After shuffling, the first point is: " << std::endl;
std::cout << "  " << shuffledDataset.col(0).t();
std::cout << "with label " << shuffledLabels[0] << "." << std::endl;

// Generate random weights, then shuffle those also.
arma::rowvec weights(dataset.n_cols, arma::fill::randu);
arma::rowvec shuffledWeights;
mlpack::ShuffleData(dataset, labels, weights, shuffledDataset, shuffledLabels,
    shuffledWeights);

std::cout << std::endl << std::endl;
std::cout << "Before shuffling with weights, the first point was: "
    << std::endl;
std::cout << "  " << dataset.col(0).t();
std::cout << "with label " << labels[0] << " and weight " << weights[0] << "."
    << std::endl;
std::cout << std::endl;
std::cout << "After shuffling with weights, the first point is: " << std::endl;
std::cout << "  " << shuffledDataset.col(0).t();
std::cout << "with label " << shuffledLabels[0] << " and weight "
    << shuffledWeights[0] << "." << std::endl;
```

---

## Distances

mlpack includes a number of distance metrics for its distance-based techniques.
These all implement the [same API](../developer/distances.md), providing one
`Evaluate()` method, and can be used with a variety of different techniques,
including:

<!-- TODO: better names for each link -->

 * [`NeighborSearch`](/src/mlpack/methods/neighbor_search/neighbor_search.hpp)
 * [`RangeSearch`](/src/mlpack/methods/range_search/range_search.hpp)
 * [`LMNN`](methods/lmnn.md)
 * [`EMST`](/src/mlpack/methods/emst/emst.hpp)
 * [`NCA`](methods/nca.md)
 * [`RANN`](/src/mlpack/methods/rann/rann.hpp)
 * [`KMeans`](/src/mlpack/methods/kmeans/kmeans.hpp)

Supported metrics:

 * [`LMetric`](#lmetric): generalized L-metric/Lp-metric, including
   Manhattan/Euclidean/Chebyshev distances
 * [`IoUDistance`](#ioudistance): intersection-over-union distance
 * [`IPMetric<KernelType>`](#ipmetrickerneltype): inner product metric (e.g.
   induced metric over a [Mercer kernel](#kernels))
 * [`MahalanobisDistance`](#mahalanobisdistance): weighted Euclidean distance
   with weights specified by a covariance matrix
 * [Implement a custom metric](../developer/distances.md)

### `LMetric`

The `LMetric` template class implements a [generalized
L-metric](https://en.wikipedia.org/wiki/Lp_space#Definition)
(L1-metric, L2-metric, etc.).  The class has two template parameters:

```
LMetric<Power, TakeRoot>
```

 * `Power` is an `int` representing the type of the metric; e.g., `2` would
   represent the L2-metric (Euclidean distance).
   - `Power` must be `1` or greater.
   - If `Power` is `INT_MAX`, the metric is the L-infinity distance (Chebyshev
     distance).

 * `TakeRoot` is a `bool` (default `true`) indicating whether the root of the
   distance should be taken.
   - If set to `false`, the metric will no longer satisfy the triangle
     inequality.

---

Several convenient typedefs are available:

 * `ManhattanDistance` (defined as `LMetric<1>`)
 * `EuclideanDistance` (defined as `LMetric<2>`)
 * `SquaredEuclideanDistance` (defined as `LMetric<2, false>`)
 * `ChebyshevDistance` (defined as `LMetric<INT_MAX>`)

---

The static `Evaluate()` method can be used to compute the distance between two
vectors.

*Note:* The vectors given to `Evaluate()` can have any type so long as the type
implements the Armadillo API (e.g. `arma::fvec`, `arma::sp_fvec`, etc.).

---

*Example usage:*

```c++
// Create two vectors: [0, 1.0, 5.0] and [1.0, 3.0, 5.0].
arma::vec a("0.0 1.0 5.0");
arma::vec b("1.0 3.0 5.0");

const double d1 = mlpack::ManhattanDistance::Evaluate(a, b);        // d1 = 3.0
const double d2 = mlpack::EuclideanDistance::Evaluate(a, b);        // d2 = 2.24
const double d3 = mlpack::SquaredEuclideanDistance::Evaluate(a, b); // d3 = 5.0
const double d4 = mlpack::ChebyshevDistance::Evaluate(a, b);        // d4 = 2.0
const double d5 = mlpack::LMetric<4>::Evaluate(a, b);               // d5 = 2.03
const double d6 = mlpack::LMetric<3, false>::Evaluate(a, b);        // d6 = 9.0

std::cout << "Manhattan distance:         " << d1 << "." << std::endl;
std::cout << "Euclidean distance:         " << d2 << "." << std::endl;
std::cout << "Squared Euclidean distance: " << d3 << "." << std::endl;
std::cout << "Chebyshev distance:         " << d4 << "." << std::endl;
std::cout << "L4-distance:                " << d5 << "." << std::endl;
std::cout << "Cubed L3-distance:          " << d6 << "." << std::endl;

// Compute the distance between two random 10-dimensional vectors in a matrix.
arma::mat m(10, 100, arma::fill::randu);

const double d7 = mlpack::EuclideanDistance::Evaluate(m.col(0), m.col(7));

std::cout << std::endl;
std::cout << "Distance between two random vectors: " << d7 << "." << std::endl;
std::cout << std::endl;

// Compute the distance between two 32-bit precision `float` vectors.
arma::fvec fa("0.0 1.0 5.0");
arma::fvec fb("1.0 3.0 5.0");

const double d8 = mlpack::EuclideanDistance::Evaluate(fa, fb); // d8 = 2.236

std::cout << "Euclidean distance (fvec): " << d8 << "." << std::endl;
```

---

### `IoUDistance`

The `IoUDistance` class implements the intersection-over-union distance metric,
a measure of the overlap between two bounding boxes related to the
[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index).

For two bounding boxes, the `IoUDistance` is computed as
`1 - (area of intersection / area of union)`.
If the bounding boxes overlap completely, the distance is 0; if they do not
overlap at all, the distance is 1.

---

The class has a boolean template parameter `UseCoordinates` that controls how
bounding boxes are specified.

 * `IoUDistance<>` (or `IoUDistance<false>`) expects bounding boxes to be
   provided to the `Evaluate()` as four-element vectors of the form
   `[x0, y0, h, w]`, where:
    - `(x0, y0)` is the lower left corner of the bounding box,
    - `h` is the height of the bounding box, and
    - `w` is the width of the bounding box.

 * `IoUDistance<true>` expects bounding boxes to be provided to the `Evaluate()`
   as four-element vectors of the form `[x0, y0, x1, y1]`, where:
    - `(x0, y0)` is the lower left corner of the bounding box, and
    - `(x1, y1)` is the upper right corner of the bounding box.

---

The static `Evaluate()` method can be used to compute the IoU distance between
two bounding boxes.

If either input vector does not have four elements, an exception will be thrown.

*Note:* The vectors given to `Evaluate()` can have any type so long as the type
implements the Armadillo API (e.g. `arma::vec`, `arma::fvec`, etc.).  The use of
sparse objects is not recommended to represent bounding boxes (as they are in
general not sparse).

---

*Example usage:*

```c++
// Create three bounding boxes by representing the lower left and size.
arma::vec bb1("0.0 0.0 3.0 5.0"); // Lower left at (0, 0), height=3, width=5.
arma::vec bb2("2.0 2.0 5.0 2.0"); // Lower left at (2, 2), height=5, width=2.
arma::vec bb3("1.0 1.0 1.5 1.0"); // Lower left at (1, 1), height=1.5, width=1.

// Represent the same three bounding boxes in lower left/upper right form.
arma::vec bb1Coord("0.0 0.0 5.0 3.0"); // Upper right is (5, 3).
arma::vec bb2Coord("2.0 2.0 4.0 7.0"); // Upper right is (4, 7).
arma::vec bb3Coord("1.0 1.0 2.0 2.5"); // Upper right is (2, 2.5).

// Compute the distance between each of the bounding boxes using the
// height/width representation.
const double d1 = mlpack::IoUDistance<>::Evaluate(bb1, bb2);
const double d2 = mlpack::IoUDistance<>::Evaluate(bb2, bb3);
const double d3 = mlpack::IoUDistance<>::Evaluate(bb1, bb3);

std::cout << "IoUDistance with width/height bounding box representations:"
    << std::endl;
std::cout << " - ll=(0, 0), h=3, w=5 and ll=(2, 2), h=5, w=2:   " << d1
    << "." << std::endl;
std::cout << " - ll=(0, 0), h=3, w=5 and ll=(1, 1), h=1.5, w=1: " << d3
    << "." << std::endl;
std::cout << " - ll=(2, 2), h=5, w=2 and ll=(1, 1), h=1.5, w=1: " << d2
    << "." << std::endl;

// Now compute the same distances with the other representation.
const double d1Coord = mlpack::IoUDistance<true>::Evaluate(bb1Coord, bb2Coord);
const double d2Coord = mlpack::IoUDistance<true>::Evaluate(bb2Coord, bb3Coord);
const double d3Coord = mlpack::IoUDistance<true>::Evaluate(bb1Coord, bb3Coord);

std::cout << "IoUDistance with two-coordinate bounding box representations:"
    << std::endl;
std::cout << "(same bounding boxes as above)" << std::endl;
std::cout << " - ll=(0, 0), ur=(5, 3) and ll=(2, 2), ur=(4, 7):   " << d1Coord
    << "." << std::endl;
std::cout << " - ll=(0, 0), ur=(5, 3) and ll=(1, 1), ur=(2, 2.5): " << d3Coord
    << "." << std::endl;
std::cout << " - ll=(2, 2), ur=(4, 7) and ll=(1, 1), ur=(2, 2.5): " << d2Coord
    << "." << std::endl;
```

---

### `IPMetric<KernelType>`

The `IPMetric<KernelType>` class implements the distance metric induced by the
given [`KernelType`](#kernels).  This computes distances in
[kernel space](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick).
Using the fact that a kernel `k(x, y)` (represented by `KernelType`) implements
an inner product in kernel space, the `IPMetric` distance is defined as

```
d(x, y) = sqrt(k(x, x) + k(y, y) - 2 k(x, y)).
```

The template parameter `KernelType` can be any of mlpack's [kernels](#kernels),
or a [custom kernel](#implement-a-custom-kernel).

This metric is used by the [FastMKS](/src/mlpack/methods/fastmks/fastmks.hpp)
method (fast max-kernel search).

---

#### Constructors and properties

 * `d = IPMetric<KernelType>()`
   - Construct a new `IPMetric` using a default-constructed `KernelType`.
   - A default constructor for `KernelType` must be available
     (e.g. `k = KernelType()`).

 * `d = IPMetric<KernelType>(kernel)`
   - Construct a new `IPMetric` using the given `kernel` (a `KernelType`
     object).
   - `kernel` is not copied; ensure that `kernel` does not go out of scope while
     `d` is in use.

 * `d = IPMetric<KernelType>(other)`
   - Copy constructor: create a new `IPMetric` from the given `IPMetric`
     `other`.
   - This copies the internally-held `KernelType`.

 * The copy operator (`d = other;`) will also copy the internally-held
   `KernelType`.

 * The internally-held `KernelType` can be accessed with `d.Kernel()`.

---

#### Distance evaluation

 * `d.Evaluate(x1, x2)`
   - Evaluate and return the distance in kernel space between two vectors `x1`
     and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API (e.g.
     `arma::vec`, `arma::sp_vec`, etc.).
   - `x1` and `x2` must be valid inputs to the `Evaluate()` function of the
     given `KernelType`.

---

*Example usage:*

```c++
// Create a few random points.
arma::vec x1(3, arma::fill::randu);
arma::vec x2(3, arma::fill::randu);
arma::vec x3(3, arma::fill::randu);

// Create a metric on the Epanechnikov kernel.
mlpack::EpanechnikovKernel ek(1.5 /* bandwidth */);
mlpack::IPMetric<mlpack::EpanechnikovKernel> ip1(ek);

// Compute distances in kernel space, and compare with kernel evaluations.
std::cout << "x1: " << x1.t();
std::cout << "x2: " << x2.t();
std::cout << "x3: " << x3.t();
std::cout << std::endl;

std::cout << "  ek(x1, x2): " << ek.Evaluate(x1, x2) << "." << std::endl;
std::cout << "  ip(x1, x2): " << ip1.Evaluate(x1, x2) << "." << std::endl;
std::cout << std::endl;

std::cout << "  ek(x2, x3): " << ek.Evaluate(x2, x3) << "." << std::endl;
std::cout << "  ip(x2, x3): " << ip1.Evaluate(x2, x3) << "." << std::endl;
std::cout << std::endl;

std::cout << "  ek(x1, x3): " << ek.Evaluate(x1, x3) << "." << std::endl;
std::cout << "  ip(x1, x3): " << ip1.Evaluate(x1, x3) << "." << std::endl;
std::cout << std::endl;

// Change the bandwidth of the kernel.
ip1.Kernel().Bandwidth(2.0);
std::cout << "With bandwidth 2.0:" << std::endl;
std::cout << "  ek(x1, x3): " << ek.Evaluate(x1, x3) << "." << std::endl;
std::cout << "  ip(x1, x3): " << ip1.Evaluate(x1, x3) << "." << std::endl;
std::cout << std::endl;

// Now create a metric on the LinearKernel.
// This one is a bit of a trick!  For the LinearKernel, the induced metric is
// exactly the Euclidean distance.
mlpack::IPMetric<mlpack::LinearKernel> ip2;

std::cout << "  Euclidean distance between x1/x2:     "
    << mlpack::EuclideanDistance::Evaluate(x1, x2) << "." << std::endl;
std::cout << "  IPMetric<LinearKernel> between x1/x2: "
    << ip2.Evaluate(x1, x2) << "." << std::endl;

// Compute the kernel space distance between two floating-point vectors.
arma::fvec fx1(10, arma::fill::randu);
arma::fvec fx2(10, arma::fill::randu);

std::cout << "IPMetric<EpanechnikovKernel> result between two random "
    << "10-dimensional 32-bit floating point vectors:" << std::endl;
std::cout << "  " << ip1.Evaluate(fx1, fx2) << "." << std::endl;
```

---

### `MahalanobisDistance`

The `MahalanobisDistance` class implements the weighted Euclidean distance known
as the
[Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance).
This distance requires an inverse covariance matrix `Q` that controls the
weighting of individual dimensions in the distance calculation.  The metric is
defined as:

```
d_Q(x, y) = sqrt((x - y)^T Q (x - y))
```

The class has two template parameters:

```
MahalanobisDistance<TakeRoot = true, MatType = arma::mat>
```

 * When `TakeRoot` is manually specified as `false`, the `sqrt()` is omitted.
   This is slightly faster, but will cause the distance to no longer satisfy the
   triangle inequality.

 * `MatType` is the matrix type used to represent `Q`, and should be a matrix
   type satisfying the Armadillo API (e.g.  `arma::mat`, `arma::fmat`).

***Notes:***

 - Many descriptions of the Mahalanobis distance use the term `C^-1` instead of
   `Q` as used here.  Ensure that the given `Q` matrix is the inverted
   covariance (you can use, e.g.,
   [`arma::pinv()`](https://arma.sourceforge.net/docs.html#pinv)).

 - Instead of using `MahalanobisDistance` directly as a distance metric for
   mlpack machine learning algorithms, it can often be faster to simply multiply
   the dataset by the equivalent transformation implied by `Q` and then use that
   modified dataset with the Euclidean distance directly.  See the example usage
   below.

---

#### Constructors and properties

 * `md = MahalanobisDistance()`
   - Create a `MahalanobisDistance` object without initializing the inverse
     covariance `Q`.
   - Call `Q()` to set the matrix before calling `Evaluate()`.

 * `md = MahalanobisDistance(dimensionality)`
   - Create a `MahalanobisDistance` where `Q` is the identity matrix of the
     given `dimensionality`.
   - This distance metric will be equivalent to the Euclidean distance.

 * `md = MahalanobisDistance(matQ)`
   - Create a `MahalanobisDistance` with the given `Q` matrix.
   - `matQ` must be positive definite and symmetric.

 * `md.Q()`
   - Access or modify the `Q` matrix.
   - For instance, to set the `Q` matrix, `md.Q() = myCustomQ;` can be used.
   - The `Q` matrix must be positive definite and symmetric.

---

#### Distance evaluation

 * `md.Evaluate(x1, x2)`
   - Evaluate and return the Mahalanobis distance between two vectors `x1` and
     `x2`.
   - `x1` and `x2` should be vector types with element type equivalent to the
     element type of `MatType` (e.g. `arma::vec`, `arma::fvec`, etc.).

---

*Example usage:*

```c++
// Create random 10-dimensional data.
arma::mat dataset(10, 100, arma::fill::randu);

// Create a positive-definite Q matrix by using a weighting matrix W such that
// Q = W^T W.
arma::mat W(10, 10, arma::fill::randu);
arma::mat Q = W.t() * W;

// Create a MahalanobisDistance object with the given Q.
mlpack::MahalanobisDistance md(std::move(Q));

std::cout << "Mahalanobis distance between points 3 and 4: "
    << md.Evaluate(dataset.col(3), dataset.col(4)) << "." << std::endl;

// Now compare the Mahalanobis distance with the Euclidean distance on the
// dataset transformed with W.  (They are the same!)
arma::mat transformedDataset = W * dataset;
std::cout << "Mahalanobis distance between points 2 and 71:           "
    << md.Evaluate(dataset.col(2), dataset.col(71)) << "." << std::endl;
std::cout << "Euclidean distance between transformed points 2 and 71: "
    << mlpack::EuclideanDistance::Evaluate(transformedDataset.col(2),
                                           transformedDataset.col(71))
    << "." << std::endl;

// Create a Mahalanobis distance for 32-bit floating point data.
arma::fmat floatDataset(20, 100, arma::fill::randn);

// Use a random diagonal matrix as Q.
arma::fmat fQ = arma::diagmat(arma::randu<arma::fvec>(20));

mlpack::MahalanobisDistance<false /* do not take square root */,
                            arma::fmat> fmd;
fmd.Q() = std::move(fQ);

const double d1 = fmd.Evaluate(floatDataset.col(3), floatDataset.col(5));
const double d2 = fmd.Evaluate(floatDataset.col(11), floatDataset.col(31));

std::cout << "Squared Mahalanobis distance on 32-bit floating point data:"
    << std::endl;
std::cout << " - Points 3 and 5:   " << d1 << "." << std::endl;
std::cout << " - Points 11 and 31: " << d2 << "." << std::endl;

// Note that an equivalent transformation matrix can be recovered from Q with
// an upper Cholesky decomposition (Q -> R.t() * R).
arma::mat recoveredW = arma::chol(md.Q(), "lower");
// A transformed dataset can be created with `(recoveredW * dataset)`.
```

---

## Distributions

<!-- TODO: link to the completed HMM documentation -->

mlpack has support for a number of different distributions, each supporting the
same API.  These can be used with, for instance, the
[`HMM`](/src/mlpack/methods/hmm/hmm.hpp) class.

 * [`DiscreteDistribution`](#discretedistribution): multidimensional categorical
   distribution (generalized Bernoulli distribution)
 * [`GaussianDistribution`](#gaussiandistribution): multidimensional Gaussian
   distribution

### `DiscreteDistribution`

`DiscreteDistribution` represents a multidimensional categorical distribution
(or generalized Bernoulli distribution) where integer-valued vectors (e.g.
`[0, 3, 4]`) are associated with specific probabilities in each dimension.

*Example:* a 3-dimensional `DiscreteDistribution` will have a specific
probability value associated with each integer value in each dimension.  So, for
the vector `[0, 3, 4]`, `P(0)` in dimension 0 could be, e.g., `0.3`, `P(3)` in
dimension 1 could be, e.g., `0.4`, and `P(4)` in dimension 2 could be, e.g.,
`0.6`.  Then, `P([0, 3, 4])` would be `0.3 * 0.4 * 0.6 = 0.072`.

---

#### Constructors

 * `d = DiscreteDistribution(numObservations)`
   - Create a one-dimensional discrete distribution with `numObservations`
     different observations in the one and only dimension.  `numObservations` is
     of type `size_t`.

 * `d = DiscreteDistribution(numObservationsVec)`
   - Create a multidimensional discrete distribution with
     `numObservationsVec.n_elem` dimensions and `numObservationsVec[i]`
     different observations in dimension `i`.
   - `numObservationsVec` is of type `arma::Col<size_t>`.

 * `d = DiscreteDistribution(probabilities)`
   - Create a multidimensional discrete distribution with the given
     probabilities.
   - `probabilities` should have type `std::vector<arma::vec>`, and
     `probabilities.size()` should be equal to the dimensionality of the
     distribution.
   - `probabilities[i]` is a vector such that `probabilities[i][j]` contains the
     probability of `j` in dimension `i`.

---

#### Access and modify properties of distribution

 * `d.Dimensionality()` returns a `size_t` indicating the number of dimensions
   in the multidimensional discrete distribution.

 * `d.Probabilities(i)` returns an `arma::vec&` containing the probabilities of
   each observation in dimension `i`.
   - `d.Probabilities(i)[j]` is the probability of `j` in dimension `i`.
   - This can be used to modify probabilities: `d.Probabilities(0)[1] = 0.7`
     sets the probability of observing the value `1` in dimension `0` to `0.7`.
   - *Note:* when setting probabilities manually, be sure that the sum of
     probabilities in a dimension is 1!

---

#### Compute probabilities of points

 * `d.Probability(observation)` returns the probability of the given
   observation as a `double`.
   - `observation` should be an `arma::vec` of size `d.Dimensionality()`.
   - `observation[i]` should take integer values between `0` and
     `d.Probabilities(i).n_elem - 1`.

 * `d.Probability(observations, probabilities)` computes the probabilities of
   many observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `d.Probability(observations.col(i))`.

 * `d.LogProbability(observation)` returns the log-probability of the given
   observation as a `double`.

 * `d.LogProbability(observations, probabilities)` computes the
   log-probabilities of many observations.

---

#### Sample from the distribution

 * `d.Random()` returns an `arma::vec` with a random sample from the
   multidimensional discrete distribution.

---

#### Fit the distribution to observations

 * `d.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `observations(j, i)` should be an integer value between `0` and the number
     of observations for dimension `i`.

 * `d.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `d`.

---

*Example usage:*

```c++
// Create a single-dimension Bernoulli distribution: P([0]) = 0.3, P([1]) = 0.7.
mlpack::DiscreteDistribution bernoulli(2);
bernoulli.Probabilities(0)[0] = 0.3;
bernoulli.Probabilities(0)[1] = 0.7;

const double p1 = bernoulli.Probability(arma::vec("0")); // p1 = 0.3.
const double p2 = bernoulli.Probability(arma::vec("1")); // p2 = 0.7.

// Create a 3-dimensional discrete distribution by specifying the probabilities
// manually.
arma::vec probDim0 = arma::vec("0.1 0.3 0.5 0.1"); // 4 possible values.
arma::vec probDim1 = arma::vec("0.7 0.3");         // 2 possible values.
arma::vec probDim2 = arma::vec("0.4 0.4 0.2");     // 3 possible values.
std::vector<arma::vec> probs { probDim0, probDim1, probDim2 };
mlpack::DiscreteDistribution d(probs);

arma::vec obs("2 0 1");
const double p3 = d.Probability(obs); // p3 = 0.5 * 0.7 * 0.4 = 0.14.

// Estimate a 10-dimensional discrete distribution.
// Each dimension takes values between 0 and 9.
arma::mat observations = arma::randi<arma::mat>(10, 1000,
    arma::distr_param(0, 9));

// Create a distribution with 10 observations in each of the 10 dimensions.
mlpack::DiscreteDistribution d2(
    arma::Col<size_t>("10 10 10 10 10 10 10 10 10 10"));
d2.Train(observations);

// Compute the probabilities of each point.
arma::vec probabilities;
d2.Probability(observations, probabilities);
std::cout << "Average probability: " << arma::mean(probabilities) << "."
    << std::endl;
```

---

### `GaussianDistribution`

`GaussianDistribution` is a standard multivariate Gaussian distribution with
parameterized mean and covariance.

---

#### Constructors

 * `g = GaussianDistribution(dimensionality)`
   - Create the distribution with the given dimensionality.
   - The distribution will have a zero mean and unit diagonal covariance matrix.

 * `g = GaussianDistribution(mean, covariance)`
   - Create the distribution with the given mean and covariance.
   - `mean` is of type `arma::vec` and should have length equal to the
     dimensionality of the distribution.
   - `covariance` is of type `arma::mat`, and should be symmetric and square,
     with rows and columns equal to the dimensionality of the distribution.

---

#### Access and modify properties of distribution

 * `g.Dimensionality()` returns the dimensionality of the distribution as a
   `size_t`.

 * `g.Mean()` returns an `arma::vec&` holding the mean of the distribution.
   This can be modified.

 * `g.Covariance()` returns a `const arma::mat&` holding the covariance of the
   distribution.  To set a new covariance, use `g.Covariance(newCov)` or
   `g.Covariance(std::move(newCov))`.

 * `g.InvCov()` returns a `const arma::mat&` holding the precomputed inverse of
   the covariance.

 * `g.LogDetCov()` returns a `double` holding the log-determinant of the
   covariance.

---

#### Compute probabilities of points

 * `g.Probability(observation)` returns the probability of the given
   observation as a `double`.
   - `observation` should be an `arma::vec` of size `d.Dimensionality()`.

 * `g.Probability(observations, probabilities)` computes the probabilities of
   many observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `g.Probability(observations.col(i))`.

 * `g.LogProbability(observation)` returns the log-probability of the given
   observation as a `double`.

 * `g.LogProbability(observations, probabilities)` computes the
   log-probabilities of many observations.

---

#### Sample from the distribution

 * `g.Random()` returns an `arma::vec` with a random sample from the
   multidimensional discrete distribution.

---

#### Fit the distribution to observations

 * `g.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.

 * `g.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `d`.

---

*Example usage:*

```c++
// Create a Gaussian distribution in 3 dimensions with zero mean and unit
// covariance.
mlpack::GaussianDistribution g(3);

// Compute the probability of the point [0, 0.5, 0.25].
const double p = g.Probability(arma::vec("0 0.5 0.25"));

// Modify the mean in dimension 0.
g.Mean()[0] = 0.5;

// Set a random covariance.
arma::mat newCov(3, 3, arma::fill::randu);
newCov *= newCov.t(); // Ensure covariance is positive semidefinite.
g.Covariance(std::move(newCov)); // Set new covariance.

// Compute the probability of the same point [0, 0.5, 0.25].
const double p2 = g.Probability(arma::vec("0 0.5 0.25"));

// Create a Gaussian distribution that is estimated from random samples in 50
// dimensions.
arma::mat samples(50, 10000, arma::fill::randn); // Normally distributed.

mlpack::GaussianDistribution g2(50);
g2.Train(samples);

// Compute the probability of all of the samples.
arma::vec probabilities;
g2.Probability(samples, probabilities);

std::cout << "Average probability is: " << arma::mean(probabilities) << "."
    << std::endl;
```

## Kernels

mlpack includes a number of Mercer kernels for its kernel-based techniques.
These all implement the [same API](../developer/kernels.md), providing one
`Evaluate()` method, and can be used with a variety of different techniques,
including:

<!-- TODO: document everything below -->

 * [`KDE`](/src/mlpack/methods/kde/kde.hpp)
 * [`MeanShift`](/src/mlpack/methods/mean_shift/mean_shift.hpp)
 * [`KernelPCA`](/src/mlpack/methods/kernel_pca/kernel_pca.hpp)
 * [`FastMKS`](/src/mlpack/methods/fastmks/fastmks.hpp)
 * [`NystroemMethod`](/src/mlpack/methods/nystroem_method/nystroem_method.hpp)

Supported kernels:

 * [`GaussianKernel`](#gaussiankernel): standard Gaussian/radial basis
   function/RBF kernel
 * [`CauchyKernel`](#cauchykernel): Cauchy kernel, with longer tails than the
   standard Gaussian kernel
 * [`CosineSimilarity`](#cosinesimilarity): dot-product vector similarity
 * [`EpanechnikovKernel`](#epanechnikovkernel): Epanechnikov kernel (parabolic),
   with zero tails
 * [`HyperbolicTangentKernel`](#hyperbolictangentkernel): hyperbolic tangent
   kernel (not positive definite)
 * [`LaplacianKernel`](#laplaciankernel): Laplacian kernel/exponential kernel
 * [`LinearKernel`](#linearkernel): linear (dot-product) kernel
 * [`PolynomialKernel`](#polynomialkernel): arbitrary-power polynomial kernel
   with offset
 * [`PSpectrumStringKernel`](#pspectrumstringkernel): kernel to compute length-p
   subsequence match counts
 * [`SphericalKernel`](#sphericalkernel): spherical/uniform/rectangular window
   kernel
 * [`TriangularKernel`](#triangularkernel): triangular kernel, with zero tails
 * [Implement a custom kernel](#implement-a-custom-kernel)

### `GaussianKernel`

The `GaussianKernel` class implements the standard [Gaussian
kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) (also called
the _radial basis function kernel_ or _RBF kernel_).

The Gaussian kernel is defined as:
`k(x1, x2) = exp(-|| x1 - x2 ||^2 / (2 * bw^2))`
where `bw` is the bandwidth parameter of the kernel.

---

#### Constructors and properties

 * `g = GaussianKernel(bw=1.0)`
   - Create a `GaussianKernel` with the given bandwidth `bw`.

 * `g.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `g.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `g.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `g.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

---

#### Other utilities

 * `g.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](methods/mean_shift.md).

 * `g.Normalizer(dimensionality)`
   - Return the
     [normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant)
     of the Gaussian kernel for points in the given dimensionality as a
     `double`.

---

*Example usage:*

```c++
// Create a Gaussian kernel with default bandwidth.
mlpack::GaussianKernel g;

// Create a Gaussian kernel with bandwidth 5.0.
mlpack::GaussianKernel g2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = g.Evaluate(x1, x2);
const double k2 = g2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Evaluate the kernel value when the distance between two points is already
// computed.
const double distance = 1.5;
const double k3 = g.Evaluate(distance);

// Change the bandwidth of the kernel to 2.5.
g.Bandwidth(2.5);
const double k4 = g.Evaluate(x1, x2);
std::cout << "Kernel value with bw=2.5: " << k4 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = g.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k5 = g.Evaluate(fx1, fx2);
const double k6 = g2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k5
    << " (bw=2.5), " << k6 << " (bw=5.0)." << std::endl;
```

### `CauchyKernel`

The `CauchyKernel` class implements the Cauchy kernel, a kernel function with a
longer tail than the Gaussian kernel, defined as:
`k(x1, x2) = 1 / (1 + (|| x1 - x2 ||^2 / bw^2))`
where `bw` is the bandwidth parameter of the kernel.

---

#### Constructors and properties

 * `c = CauchyKernel(bw=1.0)`
   - Create a `CauchyKernel` with the given bandwidth `bw`.

 * `c.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `c.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `c.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

---

*Example usage:*

```c++
// Create a Cauchy kernel with default bandwidth.
mlpack::CauchyKernel c;

// Create a Cauchy kernel with bandwidth 5.0.
mlpack::CauchyKernel c2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = c.Evaluate(x1, x2);
const double k2 = c2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Change the bandwidth of the kernel to 2.5.
c.Bandwidth(2.5);
const double k3 = c.Evaluate(x1, x2);
std::cout << "Kernel value with bw=2.5: " << k3 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = c.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k4 = c.Evaluate(fx1, fx2);
const double k5 = c2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k4
    << " (bw=2.5), " << k5 << " (bw=5.0)." << std::endl;
```

### `CosineSimilarity`

The `CosineSimilarity` class implements the dot-product cosine similarity,
defined as:
`k(x1, x2) = (x1^T x2) / (|| x1 || * || x2 ||)`.
The value of the kernel is limited to the range `[-1, 1]`.
The cosine similarity is often used in text mining tasks.

---

#### Constructor

 * `c = CosineSimilarity()`
   - Create a `CosineSimilarity` object.

***Note:*** because the `CosineSimilarity` kernel has no parameters, it is not
necessary to create an object and the `Evaluate()` function (below) can be
called statically.

---

#### Kernel evaluation

 * `c.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2` with an
     instantiated `CosineSimilarity` object.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `CosineDistance::Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2` without an
     instantiated `CosineSimilarity` object (e.g. call `Evaluate()` statically).
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

---

*Example usage:*

```c++
// Create a cosine similarity kernel.
mlpack::CosineSimilarity c;

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = c.Evaluate(x1, x2);
const double k2 = c.Evaluate(x1, x1);
const double k3 = c.Evaluate(x2, x2);
std::cout << "Cosine similarity values:" << std::endl;
std::cout << "  - k(x1, x2): " << k1 << "." << std::endl;
std::cout << "  - k(x1, x1): " << k2 << "." << std::endl;
std::cout << "  - k(x2, x2): " << k3 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix,
// using the static Evaluate() function.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = mlpack::CosineSimilarity::Evaluate(x1, r.col(i));
std::cout << "Average cosine similarity for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the cosine similarity between two sparse 32-bit floating point
// vectors.
arma::sp_fvec x3, x4;
x3.sprandu(100, 1, 0.2);
x4.sprandu(100, 1, 0.2);
const double k4 = mlpack::CosineSimilarity::Evaluate(x3, x4);
std::cout << "Cosine similarity between two random sparse 32-bit floating "
    << "point vectors: " << k4 << "." << std::endl;
```

### `EpanechnikovKernel`

The `EpanechnikovKernel` implements the
[parabolic or Epanechnikov kernel](https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use),
defined as the following function:
`k(x1, x2) = max(0, (3 / 4) * (1 - (|| x1 - x2 ||_2 / bw)^2))`,
where `bw` is the bandwidth parameter of the kernel.

The kernel takes the value `0` when `|| x1 - x2 ||_2` (the Euclidean
distance between `x1` and `x2`) is greater than or equal to `bw`.

---

#### Constructors and properties

 * `e = EpanechnikovKernel(bw=1.0)`
   - Create an `EpanechnikovKernel` with the given bandwidth `bw`.

 * `e.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `e.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `e.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `e.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

---

#### Other utilities

 * `e.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](methods/mean_shift.md).

 * `e.Normalizer(dimensionality)`
   - Return the
     [normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant)
     of the Epanechnikov kernel for points in the given dimensionality as a
     `double`.

---

*Example usage:*

```c++
// Create an Epanechnikov kernel with default bandwidth.
mlpack::EpanechnikovKernel e;

// Create an Epanechnikov kernel with bandwidth 5.0.
mlpack::EpanechnikovKernel e2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = e.Evaluate(x1, x2);
const double k2 = e2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Evaluate the kernel value when the distance between two points is already
// computed.
const double distance = 1.5;
const double k3 = e.Evaluate(distance);

// Change the bandwidth of the kernel to 2.5.
e.Bandwidth(2.5);
const double k4 = e.Evaluate(x1, x2);
std::cout << "Kernel value with bw=2.5: " << k4 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = e.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k5 = e.Evaluate(fx1, fx2);
const double k6 = e2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k5
    << " (bw=2.5), " << k6 << " (bw=5.0)." << std::endl;
```

### `HyperbolicTangentKernel`

The `HyperbolicTangentKernel` implements the
[hyperbolic tangent kernel](https://en.wikipedia.org/wiki/Support_vector_machine#Nonlinear_kernels),
which is defined by the following equation:
`f(x1, x2) = tanh(s * (x1^T x2) + t)`
where `s` is the scale parameter and `t` is the offset parameter.

The hyperbolic tangent kernel is *not a positive definite Mercer kernel* and
thus does not satisfy the theoretical requirements of many kernel methods.  See
[this discussion](https://stats.stackexchange.com/questions/199620/on-the-properties-of-hyperbolic-tangent-kernel)
for more details.  In practice, for many kernel methods, it may still provide
compelling results despite this theoretical limitation.

---

#### Constructors and properties

 * `h = HyperbolicTangentKernel(s=1.0, t=0.0)`
   - Create a `HyperbolicTangentKernel` with the given scale factor `s` and the
     given offset `t`.

 * `h.Scale()` returns the scale factor of the kernel as a `double`.
   - To set the scale parameter, use `h.Scale(scale)`.

 * `h.Offset()` returns the offset parameter of the kernel as a `double`.
   - To set the offset parameter, use `h.Offset(offset)`.

---

#### Kernel evaluation

 * `h.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

---

*Example usage:*

```c++
// Create a hyperbolic tangent kernel with default scale and offset.
mlpack::HyperbolicTangentKernel h;

// Create a hyperbolic tangent kernel with scale 2.0 and offset 1.0.
mlpack::HyperbolicTangentKernel h2(2.0, 1.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = h.Evaluate(x1, x2);
const double k2 = h2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (s=1.0, t=0.0), " << k2
    << " (s=2.0, t=1.0)." << std::endl;

// Change the scale and offset of the kernel.
h.Scale(2.5);
h.Offset(-1.0);
const double k3 = h.Evaluate(x1, x2);
std::cout << "Kernel value with s=2.5, t=-1.0: " << k3 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = h.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k4 = h.Evaluate(fx1, fx2);
const double k5 = h2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k4
    << " (s=2.5, t=-1.0), " << k5 << " (s=2.0, t=1.0)." << std::endl;
```

### `LaplacianKernel`

The `LaplacianKernel` class implements the Laplacian kernel, also known as the
exponential kernel, defined by the following equation:
`k(x1, x2) = exp(-|| x1 - x2 || / bw)`
where `bw` is the bandwidth parameter.

---

#### Constructors and properties

 * `l = LaplacianKernel(bw=1.0)`
   - Create a `LaplacianKernel` with the given bandwidth `bw`.

 * `l.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `l.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `l.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `l.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

---

#### Other utilities

 * `l.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](methods/mean_shift.md).

---

*Example usage:*

```c++
// Create a Laplacian kernel with default bandwidth.
mlpack::LaplacianKernel l;

// Create a Laplacian kernel with bandwidth 5.0.
mlpack::LaplacianKernel l2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = l.Evaluate(x1, x2);
const double k2 = l2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Evaluate the kernel value when the distance between two points is already
// computed.
const double distance = 1.5;
const double k3 = l.Evaluate(distance);

// Change the bandwidth of the kernel to 2.5.
l.Bandwidth(2.5);
const double k4 = l.Evaluate(x1, x2);
std::cout << "Kernel value with bw=2.5: " << k4 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = l.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k5 = l.Evaluate(fx1, fx2);
const double k6 = l2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k5
    << " (bw=2.5), " << k6 << " (bw=5.0)." << std::endl;
```

### `LinearKernel`

The `LinearKernel` class implements the simple linear dot product kernel,
defined by the following equation:
`k(x1, x2) = x1^T x2`.

The use of the linear kernel for kernel methods generally results in the
non-kernelized version of the algorithm; for instance, a kernel support
vector machine using the linear kernel amounts to a [linear
SVM](methods/linear_svm.md).

---

#### Constructor

 * `l = LinearKernel()`
   - Create a `LinearKernel` object.

***Note:*** because the `LinearKernel` kernel has no parameters, it is not
necessary to create an object and the `Evaluate()` function (below) can be
called statically.

---

#### Kernel evaluation

 * `l.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2` with an
     instantiated `LinearKernel` object.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `LinearKernel::Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2` without an
     instantiated `LinearKernel` object (e.g. call `Evaluate()` statically).
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

---

*Example usage:*

```c++
// Create a linear kernel.
mlpack::LinearKernel l;

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = l.Evaluate(x1, x2); // Identical to arma::dot(x1, x2).
const double k2 = l.Evaluate(x1, x1);
const double k3 = l.Evaluate(x2, x2);
std::cout << "Linear kernel values:" << std::endl;
std::cout << "  - k(x1, x2): " << k1 << "." << std::endl;
std::cout << "  - k(x1, x1): " << k2 << "." << std::endl;
std::cout << "  - k(x2, x2): " << k3 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix,
// using the static Evaluate() function.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = mlpack::LinearKernel::Evaluate(x1, r.col(i));
std::cout << "Average linear kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the cosine similarity between two sparse 32-bit floating point
// vectors.
arma::sp_fvec x3, x4;
x3.sprandu(100, 1, 0.2);
x4.sprandu(100, 1, 0.2);
const double k4 = mlpack::LinearKernel::Evaluate(x3, x4);
std::cout << "Linear kernel value between two random sparse 32-bit floating "
    << "point vectors: " << k4 << "." << std::endl;
```

### `PolynomialKernel`

The `PolynomialKernel` class implements the standard
[polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel), which is
defined by the following equation:
`k(x1, x2) = (x1^T x2 + t)^d`
where `d` is the degree of the polynomial and `t` is the offset.

The use of the polynomial kernel has a similar effect to the use of polynomial
(interaction) features in standard machine learning methods.

---

#### Constructors and properties

 * `p = PolynomialKernel(d=2.0, t=0.0)`
   - Create a `PolynomialKernel` with the given degree `d` and given offset `t`.

 * `p.Degree()` returns the degree of the kernel as a `double`.
   - To set the degree, use `p.Degree(newDegree)`.

 * `p.Offset()` returns the offset of the kernel as a `double`.
   - To set the offset, use `p.Offset(newOffset)`.

---

#### Kernel evaluation

 * `p.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

---

*Example usage:*

```c++
// Create a polynomial kernel with default degree (2) and offset (0).
mlpack::PolynomialKernel p;

// Create a polynomial kernel with degree 3.0 and offset -1.0.
mlpack::PolynomialKernel p2(3.0, -1.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = p.Evaluate(x1, x2);
const double k2 = p2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Change the degree of the kernel to 2.5 and the offset to 1.0.
p.Degree(2.5);
p.Offset(1.0);
const double k3 = p.Evaluate(x1, x2);
std::cout << "Kernel value with d=2.5, t=1.0: " << k3 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = p.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k4 = p.Evaluate(fx1, fx2);
const double k5 = p2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k4
    << " (d=2.5, t=1.0), " << k5 << " (d=3.0, t=-1.0)." << std::endl;
```

### `PSpectrumStringKernel`

The `PSpectrumStringKernel` class implements the length-`p` string spectrum
kernel, proposed by
[Leslie, Eskin, and Noble (pdf)](http://psb.stanford.edu/psb-online/proceedings/psb02/leslie.pdf).
The kernel finds the contiguous subsequence match count between two strings.

Due to mlpack's use of Armadillo, which requires that all matrix data be
numeric, this class operates by internally storing all strings, and passing in
numeric vectors such as `[0 1]` that reference string index `1` in dataset index
`0`.  In turn, this means that the data points given to the
`PSpectrumStringKernel` are simply IDs and have no geometric meaning.

---

#### Constructors and properties

 * `p = PSpectrumStringKernel(datasets, p)`
    - Create a `PSpectrumStringKernel` on the given set of string datasets,
      using the given substring length `p`.
    - `datasets` should have type `std::vector<std::vector<std::string>>`, and
      contains a list of datasets, each of which is made up of a list of
      strings.
      * Multiple datasets are supported for the case where, e.g., there are
        multiple files containing different sets of strings.
    - So, e.g., `datasets[0]` represents the `0`th dataset, and `datasets[0][1]`
      is the string with index `1` inside the `0`th dataset.
    - `p` (a `size_t`) is the length of substring to use for the kernel, and
      must be greater than `0`.
    - The constructor will build counts of all substrings in the dataset, and
      for large data may be computationally intensive.

 * `p.P()` returns the substring length `p` of the kernel as a `size_t`.
    - The value of `p` cannot be changed once the object is constructed.

 * `p.Counts()` returns a `std::vector<std::vector<std::map<std::string, int>>>`
   that maps a substring to the number of times it appears in the original
   string.  So, given a substring length of `5`, `p.Counts()[0][1]["hello"]`
   would be the number of times the substring `hello` appears in the string with
   index `1` in the dataset with index `0`.

---

#### Kernel evaluation

 * `p.Evaluate(x1, x2)`
   - Compute the kernel value between two index vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`, `arma::uvec`, etc.).
   - `x1` and `x2` do not contain string data directly, but instead are each
     length-2 vectors that represent the index of the datasets and strings to be
     compared in the `datasets` object that was passed to the constructor.
   - So, e.g., if `x1 = [0, 0]` and `x2 = [1, 1]`, then the first string from
     the first dataset will be compared with the second string from the second
     dataset.

---

*Example usage:*

```c++
// Create two example datasets:
//      ["hello", "goodbye", "package"],
//      ["mlpack", "is", "really", "great"]
std::vector<std::vector<std::string>> datasets;
datasets.push_back({ "hello", "goodbye", "package" });
datasets.push_back({ "mlpack", "is", "really", "great" });

// Create a p-spectrum string kernel with a substring length of 2,
// and another with a substring length of 3.
mlpack::PSpectrumStringKernel p(datasets, 2);
mlpack::PSpectrumStringKernel p2(datasets, 3);

// Evaluate the kernel value between "mlpack" and "package".
arma::uvec x1("1 0"); // "mlpack": dataset 1, string 0.
arma::uvec x2("0 2"); // "package": dataset 0, string 2.
const double k1 = p.Evaluate(x1, x2);
const double k2 = p2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (p=2), " << k2 << " (p=3)."
    << std::endl;
```

### `SphericalKernel`

The `SphericalKernel` class implements the simple spherical kernel, also known
as the uniform kernel, or rectangular window kernel.  The value of the
`SphericalKernel` is `1` when the Euclidean distance between two points `x1` and
`x2` is less than the bandwidth `bw`, and `0` otherwise:
`k(x1, x2) = 1(|| x1 - x2 || <= bw)`.

---

#### Constructors and properties

 * `s = SphericalKernel(bw=1.0)`
   - Create a `SphericalKernel` with the given bandwidth `bw`.

 * `s.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `s.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `s.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `s.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

---

#### Other utilities

 * `s.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](methods/mean_shift.md).

 * `s.Normalizer(dimensionality)`
   - Return the
     [normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant)
     of the spherical kernel for points in the given dimensionality as a
     `double`.

---

*Example usage:*

```c++
// Create a spherical kernel with default bandwidth.
mlpack::SphericalKernel s;

// Create a spherical kernel with bandwidth 5.0.
mlpack::SphericalKernel s2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 2.5");
arma::vec x2("2.5 1.0 0.5");
const double k1 = s.Evaluate(x1, x2);
const double k2 = s2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Evaluate the kernel value when the distance between two points is already
// computed.
const double distance = 0.9;
const double k3 = s.Evaluate(distance);

// Change the bandwidth of the kernel to 3.0.
s.Bandwidth(3.0);
const double k4 = s.Evaluate(x1, x2);
std::cout << "Kernel value with bw=3.0: " << k4 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix, using
// a kernel bandwidth of 2.5.
s.Bandwidth(2.5);
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = s.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 2.5");
arma::fvec fx2("2.5 1.0 0.5");
const double k5 = s.Evaluate(fx1, fx2);
const double k6 = s2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k5
    << " (bw=2.5), " << k6 << " (bw=5.0)." << std::endl;
```

### `TriangularKernel`

The `TriangularKernel` class implements the
[simple triangular kernel](https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use),
defined by the following equation:
`k(x1, x2) = max(0, 1 - || x1 - x2 || / bw)`
where `bw` is the bandwidth of the kernel.

---

#### Constructors and properties

 * `t = TriangularKernel(bw=1.0)`
   - Create a `TriangularKernel` with the given bandwidth `bw`.

 * `t.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `t.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `t.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `t.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

---

#### Other utilities

 * `t.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](methods/mean_shift.md).

---

*Example usage:*

```c++
// Create a triangular kernel with default bandwidth.
mlpack::TriangularKernel t;

// Create a triangular kernel with bandwidth 5.0.
mlpack::TriangularKernel t2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = t.Evaluate(x1, x2);
const double k2 = t2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Evaluate the kernel value when the distance between two points is already
// computed.
const double distance = 0.75;
const double k3 = t.Evaluate(distance);

// Change the bandwidth of the kernel to 2.5.
t.Bandwidth(2.5);
const double k4 = t.Evaluate(x1, x2);
std::cout << "Kernel value with bw=2.5: " << k4 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = t.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k5 = t.Evaluate(fx1, fx2);
const double k6 = t2.Evaluate(fx1, fx2);
std::cout << "Kernel values between two floating-point vectors: " << k5
    << " (bw=2.5), " << k6 << " (bw=5.0)." << std::endl;
```

### Implement a custom kernel

mlpack supports custom kernels, so long as they implement an appropriate
`Evaluate()` function.

See [The KernelType Policy in mlpack](../developer/kernels.md) for more
information.
