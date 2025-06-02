# Kernels

mlpack includes a number of Mercer kernels for its kernel-based techniques.
These all implement the [same API](../../developer/kernels.md), providing one
`Evaluate()` method.  mlpack provides a number of supported kernels:

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

These kernels can then be used in a number of machine learning algorithms that
mlpack provides:

<!-- TODO: document everything below -->

 * [`KDE`](/src/mlpack/methods/kde/kde.hpp)
 * [`MeanShift`](/src/mlpack/methods/mean_shift/mean_shift.hpp)
 * [`KernelPCA`](/src/mlpack/methods/kernel_pca/kernel_pca.hpp)
 * [`FastMKS`](/src/mlpack/methods/fastmks/fastmks.hpp)
 * [`NystroemMethod`](/src/mlpack/methods/nystroem_method/nystroem_method.hpp)

## `GaussianKernel`

The `GaussianKernel` class implements the standard [Gaussian
kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) (also called
the _radial basis function kernel_ or _RBF kernel_).

The Gaussian kernel is defined as:
`k(x1, x2) = exp(-|| x1 - x2 ||^2 / (2 * bw^2))`
where `bw` is the bandwidth parameter of the kernel.

### Constructors and properties

 * `g = GaussianKernel(bw=1.0)`
   - Create a `GaussianKernel` with the given bandwidth `bw`.

 * `g.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `g.Bandwidth(newBandwidth)`.

### Kernel evaluation

 * `g.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `g.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

### Other utilities

 * `g.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](../methods/mean_shift.md).

 * `g.Normalizer(dimensionality)`
   - Return the
     [normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant)
     of the Gaussian kernel for points in the given dimensionality as a
     `double`.

### Example usage

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

## `CauchyKernel`

The `CauchyKernel` class implements the Cauchy kernel, a kernel function with a
longer tail than the Gaussian kernel, defined as:
`k(x1, x2) = 1 / (1 + (|| x1 - x2 ||^2 / bw^2))`
where `bw` is the bandwidth parameter of the kernel.

### Constructors and properties

 * `c = CauchyKernel(bw=1.0)`
   - Create a `CauchyKernel` with the given bandwidth `bw`.

 * `c.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `c.Bandwidth(newBandwidth)`.

### Kernel evaluation

 * `c.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

### Example usage

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

## `CosineSimilarity`

The `CosineSimilarity` class implements the dot-product cosine similarity,
defined as:
`k(x1, x2) = (x1^T x2) / (|| x1 || * || x2 ||)`.
The value of the kernel is limited to the range `[-1, 1]`.
The cosine similarity is often used in text mining tasks.

### Constructor

 * `c = CosineSimilarity()`
   - Create a `CosineSimilarity` object.

***Note:*** because the `CosineSimilarity` kernel has no parameters, it is not
necessary to create an object and the `Evaluate()` function (below) can be
called statically.

### Kernel evaluation

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

### Example usage

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

## `EpanechnikovKernel`

The `EpanechnikovKernel` implements the
[parabolic or Epanechnikov kernel](https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use),
defined as the following function:
`k(x1, x2) = max(0, (3 / 4) * (1 - (|| x1 - x2 ||_2 / bw)^2))`,
where `bw` is the bandwidth parameter of the kernel.

The kernel takes the value `0` when `|| x1 - x2 ||_2` (the Euclidean
distance between `x1` and `x2`) is greater than or equal to `bw`.

### Constructors and properties

 * `e = EpanechnikovKernel(bw=1.0)`
   - Create an `EpanechnikovKernel` with the given bandwidth `bw`.

 * `e.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `e.Bandwidth(newBandwidth)`.

### Kernel evaluation

 * `e.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `e.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

### Other utilities

 * `e.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](../methods/mean_shift.md).

 * `e.Normalizer(dimensionality)`
   - Return the
     [normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant)
     of the Epanechnikov kernel for points in the given dimensionality as a
     `double`.

### Example usage

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

## `HyperbolicTangentKernel`

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

### Constructors and properties

 * `h = HyperbolicTangentKernel(s=1.0, t=0.0)`
   - Create a `HyperbolicTangentKernel` with the given scale factor `s` and the
     given offset `t`.

 * `h.Scale()` returns the scale factor of the kernel as a `double`.
   - To set the scale parameter, use `h.Scale(scale)`.

 * `h.Offset()` returns the offset parameter of the kernel as a `double`.
   - To set the offset parameter, use `h.Offset(offset)`.

### Kernel evaluation

 * `h.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

### Example usage

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

## `LaplacianKernel`

The `LaplacianKernel` class implements the Laplacian kernel, also known as the
exponential kernel, defined by the following equation:
`k(x1, x2) = exp(-|| x1 - x2 || / bw)`
where `bw` is the bandwidth parameter.

### Constructors and properties

 * `l = LaplacianKernel(bw=1.0)`
   - Create a `LaplacianKernel` with the given bandwidth `bw`.

 * `l.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `l.Bandwidth(newBandwidth)`.

### Kernel evaluation

 * `l.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `l.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

### Other utilities

 * `l.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](../methods/mean_shift.md).

### Example usage

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

## `LinearKernel`

The `LinearKernel` class implements the simple linear dot product kernel,
defined by the following equation:
`k(x1, x2) = x1^T x2`.

The use of the linear kernel for kernel methods generally results in the
non-kernelized version of the algorithm; for instance, a kernel support
vector machine using the linear kernel amounts to a [linear
SVM](../methods/linear_svm.md).

### Constructor

 * `l = LinearKernel()`
   - Create a `LinearKernel` object.

***Note:*** because the `LinearKernel` kernel has no parameters, it is not
necessary to create an object and the `Evaluate()` function (below) can be
called statically.

### Kernel evaluation

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

### Example usage

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

## `PolynomialKernel`

The `PolynomialKernel` class implements the standard
[polynomial kernel](https://en.wikipedia.org/wiki/Polynomial_kernel), which is
defined by the following equation:
`k(x1, x2) = (x1^T x2 + t)^d`
where `d` is the degree of the polynomial and `t` is the offset.

The use of the polynomial kernel has a similar effect to the use of polynomial
(interaction) features in standard machine learning methods.

### Constructors and properties

 * `p = PolynomialKernel(d=2.0, t=0.0)`
   - Create a `PolynomialKernel` with the given degree `d` and given offset `t`.

 * `p.Degree()` returns the degree of the kernel as a `double`.
   - To set the degree, use `p.Degree(newDegree)`.

 * `p.Offset()` returns the offset of the kernel as a `double`.
   - To set the offset, use `p.Offset(newOffset)`.

### Kernel evaluation

 * `p.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

### Example usage

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

## `PSpectrumStringKernel`

The `PSpectrumStringKernel` class implements the length-`p` string spectrum
kernel, proposed by
[Leslie, Eskin, and Noble (pdf)](http://psb.stanford.edu/psb-online/proceedings/psb02/leslie.pdf).
The kernel finds the contiguous subsequence match count between two strings.

Due to mlpack's use of Armadillo, which requires that all matrix data be
numeric, this class operates by internally storing all strings, and passing in
numeric vectors such as `[0 1]` that reference string index `1` in dataset index
`0`.  In turn, this means that the data points given to the
`PSpectrumStringKernel` are simply IDs and have no geometric meaning.

### Constructors and properties

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

### Kernel evaluation

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

### Example usage

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

## `SphericalKernel`

The `SphericalKernel` class implements the simple spherical kernel, also known
as the uniform kernel, or rectangular window kernel.  The value of the
`SphericalKernel` is `1` when the Euclidean distance between two points `x1` and
`x2` is less than the bandwidth `bw`, and `0` otherwise:
`k(x1, x2) = 1(|| x1 - x2 || <= bw)`.

### Constructors and properties

 * `s = SphericalKernel(bw=1.0)`
   - Create a `SphericalKernel` with the given bandwidth `bw`.

 * `s.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `s.Bandwidth(newBandwidth)`.

### Kernel evaluation

 * `s.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `s.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

### Other utilities

 * `s.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](../methods/mean_shift.md).

 * `s.Normalizer(dimensionality)`
   - Return the
     [normalizing constant](https://en.wikipedia.org/wiki/Normalizing_constant)
     of the spherical kernel for points in the given dimensionality as a
     `double`.

### Example usage

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

## `TriangularKernel`

The `TriangularKernel` class implements the
[simple triangular kernel](https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use),
defined by the following equation:
`k(x1, x2) = max(0, 1 - || x1 - x2 || / bw)`
where `bw` is the bandwidth of the kernel.

### Constructors and properties

 * `t = TriangularKernel(bw=1.0)`
   - Create a `TriangularKernel` with the given bandwidth `bw`.

 * `t.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `t.Bandwidth(newBandwidth)`.

### Kernel evaluation

 * `t.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `t.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

### Other utilities

 * `t.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.  This is used
     by [`MeanShift`](../methods/mean_shift.md).

### Example usage

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

## Implement a custom kernel

mlpack supports custom kernels, so long as they implement an appropriate
`Evaluate()` function.

See [The KernelType Policy in mlpack](../../developer/kernels.md) for more
information.
