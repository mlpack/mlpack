# Distances

mlpack includes a number of distance metrics for its distance-based techniques.
These all implement the [same API](../../developer/distances.md), providing one
`Evaluate()` method.  mlpack provides a number of supported distance metrics:

 * [`LMetric`](#lmetric): generalized L-metric/Lp-metric, including
   Manhattan/Euclidean/Chebyshev distances
 * [`IoUDistance`](#ioudistance): intersection-over-union distance
 * [`IPMetric<KernelType>`](#ipmetrickerneltype): inner product metric (e.g.
   induced metric over a [Mercer kernel](kernels.md))
 * [`MahalanobisDistance`](#mahalanobisdistance): weighted Euclidean distance
   with weights specified by a covariance matrix
 * [Implement a custom distance metric](../../developer/distances.md)

---

These distance metrics can be used with a variety of different techniques,
including:

<!-- TODO: better names for each link -->

 * [`NeighborSearch`](/src/mlpack/methods/neighbor_search/neighbor_search.hpp)
 * [`RangeSearch`](/src/mlpack/methods/range_search/range_search.hpp)
 * [`LMNN`](../methods/lmnn.md)
 * [`EMST`](/src/mlpack/methods/emst/emst.hpp)
 * [`NCA`](../methods/nca.md)
 * [`RANN`](/src/mlpack/methods/rann/rann.hpp)
 * [`KMeans`](/src/mlpack/methods/kmeans/kmeans.hpp)

## `LMetric`

The `LMetric` template class implements a [generalized
L-metric](https://en.wikipedia.org/wiki/Lp_space#Preliminaries)
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

## `IoUDistance`

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

## `IPMetric<KernelType>`

The `IPMetric<KernelType>` class implements the distance metric induced by the
given [`KernelType`](kernels.md).  This computes distances in
[kernel space](https://en.wikipedia.org/wiki/Kernel_method#Mathematics:_the_kernel_trick).
Using the fact that a kernel `k(x, y)` (represented by `KernelType`) implements
an inner product in kernel space, the `IPMetric` distance is defined as

```
d(x, y) = sqrt(k(x, x) + k(y, y) - 2 k(x, y)).
```

The template parameter `KernelType` can be any of mlpack's
[kernels](kernels.md), or a
[custom kernel](kernels.md#implement-a-custom-kernel).

This metric is used by the [FastMKS](/src/mlpack/methods/fastmks/fastmks.hpp)
method (fast max-kernel search).

### Constructors and properties

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

### Distance evaluation

 * `d.Evaluate(x1, x2)`
   - Evaluate and return the distance in kernel space between two vectors `x1`
     and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API (e.g.
     `arma::vec`, `arma::sp_vec`, etc.).
   - `x1` and `x2` must be valid inputs to the `Evaluate()` function of the
     given `KernelType`.

### Example usage

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

## `MahalanobisDistance`

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

### Constructors and properties

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

### Distance evaluation

 * `md.Evaluate(x1, x2)`
   - Evaluate and return the Mahalanobis distance between two vectors `x1` and
     `x2`.
   - `x1` and `x2` should be vector types with element type equivalent to the
     element type of `MatType` (e.g. `arma::vec`, `arma::fvec`, etc.).

### Example usage

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
