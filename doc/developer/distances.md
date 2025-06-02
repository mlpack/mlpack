# The DistanceType policy in mlpack

Many machine learning methods operate with some sort of distance metric, and
often, this distance metric can be any arbitrary distance metric.  For instance,
consider the problem of nearest neighbor search; one can find the nearest
neighbor of a point with respect to the standard Euclidean distance, or the
Manhattan (city-block) distance.  The actual search techniques, though, remain
the same.  And this is true of many machine learning methods: the specific
distance metric that is used can be any valid distance metric.

mlpack algorithms, when relevant, allow the use of an arbitrary metric via the
use of the `DistanceType` template parameter.  Any distance metric passed as a
`DistanceType` template parameter will need to have

 - an `Evaluate()` function
 - a default constructor.

The signature of the `Evaluate()` function is straightforward:

```
template<typename VecTypeA, typename VecTypeB>
double Evaluate(const VecTypeA& a, const VecTypeB& b);
```

The function takes two vector arguments, `a` and `b`, and returns a `double`
that is the evaluation of the distance metric between the two arguments.  So,
for a particular distance metric `d`, the `Evaluate()` function should return
`d(a, b)`.

The arguments `a` and `b`, of types `VecTypeA` and `VecTypeB`, respectively,
will be an Armadillo-like vector type (usually `arma::vec`, `arma::sp_vec`, or
similar).  In general it should be valid to assume that `VecTypeA` is a class
with the same API as `arma::vec`.

Note that for distance metrics that do not hold any state, the `Evaluate()`
method can be marked as `static`.

Overall, the `DistanceType` template policy is quite simple (much like the
[KernelType policy](kernels.md)).  Below is an example distance metric class,
which implements the L2 distance:

```c++
class ExampleDistance
{
 public:
  // Default constructor is required.
  ExampleDistance() { }

  // The example metric holds no state, so we can mark Evaluate() as static.
  template<typename VecTypeA, typename VecTypeB>
  static double Evaluate(const VecTypeA& a, const VecTypeB& b)
  {
    // Return the L2 norm of the difference between the points, which is the
    // same as the L2 distance.
    return arma::norm(a - b);
  }
};
```

Then, this distance metric can easily be used inside of other mlpack algorithms.
For example, the code below runs range search on a random dataset with the
`ExampleDistance`, by instantiating a `RangeSearch` object that uses the
`ExampleDistance` with ball trees.  Then, the number of results are printed.
The `RangeSearch` class takes three template parameters: `DistanceType`,
`MatType`, and `TreeType`.

```c++
// Create a random dataset with 10 dimensions and 5000 points.
arma::mat data = arma::randu<arma::mat>(10, 5000);

// Instantiate the RangeSearch object with the ExampleDistance.
mlpack::RangeSearch<ExampleDistance, arma::mat, mlpack::BallTree> rs(data);

// These vectors will store the results.
std::vector<std::vector<size_t>> neighbors;
std::vector<std::vector<double>> distances;

// Create a random 10-dimensional query point.
arma::vec query = arma::randu<arma::vec>(10);

// Find those points with distance (according to ExampleDistance) between 1
// and 2 from the query point.
rs.Search(query, mlpack::Range(1.0, 2.0), neighbors, distances);

// Now, print the number of points inside the desired range.  We know that
// neighbors and distances will have length 1, since there was only one query
// point.
std::cout << neighbors[0].size() << " points within the range [1.0, 2.0] of the"
    << " query point!" << std::endl;
```

mlpack comes with a number of pre-written distance metrics that satisfy the
`DistanceType` policy:

 - [`ManhattanDistance`](../user/core/distances.md#lmetric)
 - [`EuclideanDistance`](../user/core/distances.md#lmetric)
 - [`ChebyshevDistance`](../user/core/distances.md#lmetric)
 - [`MahalanobisDistance`](../user/core/distances.md#mahalanobisdistance)
 - [`LMetric`](../user/core/distances.md#lmetric) (for arbitrary L-metrics)
 - [`IPMetric`](../user/core/distances.md#ipmetrickerneltype) (requires a
   [KernelType](kernels.md) parameter)
 - [`IoUDistance`](../user/core/distances.md#ioudistance)
