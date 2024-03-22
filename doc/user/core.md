# mlpack core class documentation

Underlying the implementations of [mlpack's machine learning
algorithms](index.md#mlpack-algorithm-documentation) are mlpack core support
classes, each of which are documented on this page.

 * [Core math utilities](#core-math-utilities): utility classes for mathematical
   purposes
 * [Distributions](#distributions): probability distributions
 * [Metrics](#metrics): distance metrics for geometric algorithms
 * [Kernels](#kernels): Mercer kernels for kernel-based algorithms

## Core math utilities

mlpack provides a number of additional mathematical utility classes and
functions on top of Armadillo.

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

<!-- TODO: do something with MakeAlias(); but it needs to be refactored first
-->

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

 * [`RangeSearch`](range_search.md)
 * [mlpack trees](#trees)

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
<img src="img/favicons-matrix.png" alt="four favicons each as a column in a matrix, unintelligible">
</center>

*After*:

<center>
<img src="img/favicons-blocks.png" alt="four favicons each as a block in a larger image, much better">
</center>

---

#### See also

 * [Loading and saving image data](load_save.md#image-data)
 * [`SparseAutoencoder`](sparse_autoencoder.md)

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
   [`GammaDistribution`](#gammadistribution) class.

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

## Distributions

mlpack has support for a number of different distributions, each supporting the
same API.  These can be used with, for instance, the [`HMM`](hmm.md) class.

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

## Metrics

mlpack includes a number of distance metrics for its distance-based techniques.
These all implement the [same API](../developer/metrics.md), providing one
`Evaluate()` method, and can be used with a variety of different techniques,
including:

<!-- TODO: better names for each link -->

 * [`NeighborSearch`](neighbor_search.md)
 * [`RangeSearch`](range_search.md)
 * [`LMNN`](lmnn.md)
 * [`EMST`](emst.md)
 * [`NCA`](nca.md)
 * [`RANN`](rann.md)
 * [`KMeans`](kmeans.md)

Supported metrics:

 * [`LMetric`](#lmetric): generalized L-metric/Lp-metric, including
   Manhattan/Euclidean/Chebyshev distances
 * [Implement a custom metric](../developer/metrics.md)

### `LMetric`

The `LMetric` template class implements a [generalized
L-metric](https://en.wikipedia.org/wiki/Lp_space#Definition)
(L1-metric, L2-metric, etc.).  The class has two template parameters:

```c++
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

## Kernels

mlpack includes a number of Mercer kernels for its kernel-based techniques.
These all implement the [same API](../developer/kernels.md), providing one
`Evaluate()` method, and can be used with a variety of different techniques,
including:

<!-- TODO: better names for links below -->

 * [`KDE`](kde.md)
 * [`MeanShift`](mean_shift.md)
 * [`KernelPCA`](kernel_pca.md)
 * [`FastMKS`](fastmks.md)
 * [`NystroemMethod`](nystroem_method.md)

Supported kernels:

 * [`GaussianKernel`](#gaussiankernel): standard Gaussian/radial basis
   function/RBF kernel
 * [Implement a custom kernel](../developer/kernels.md)

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
     to the distance between two points, evaluated at `distance`.

 * `g.Normalizer(dimensionality)`
   - Return the [normalizing
     constant](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) of
     the Gaussian kernel for points in the given dimensionality as a `double`.

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
```
