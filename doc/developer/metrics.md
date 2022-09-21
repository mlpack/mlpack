# The MetricType policy in mlpack

Many machine learning methods operate with some sort of metric, and often, this
metric can be any arbitrary metric.  For instance, consider the problem of
nearest neighbor search; one can find the nearest neighbor of a point with
respect to the standard Euclidean distance, or the Manhattan (city-block)
distance.  The actual search techniques, though, remain the same.  And this is
true of many machine learning methods: the specific metric that is used can be
any valid metric.

mlpack algorithms, when relevant, allow the use of an arbitrary metric via the
use of the `MetricType` template parameter.  Any metric passed as a `MetricType`
template parameter will need to have

 - an `Evaluate()` function
 - a default constructor.

The signature of the `Evaluate()` function is straightforward:

```c++
template<typename VecTypeA, typename VecTypeB>
double Evaluate(const VecTypeA& a, const VecTypeB& b);
```

The function takes two vector arguments, `a` and `b`, and returns a `double`
that is the evaluation of the metric between the two arguments.  So, for a
particular metric `d`, the `Evaluate()` function should return `d(a, b)`.

The arguments `a` and `b`, of types `VecTypeA` and `VecTypeB`, respectively,
will be an Armadillo-like vector type (usually `arma::vec`, `arma::sp_vec`, or
similar).  In general it should be valid to assume that `VecTypeA` is a class
with the same API as `arma::vec`.

Note that for metrics that do not hold any state, the `Evaluate()` method can
be marked as `static`.

Overall, the `MetricType` template policy is quite simple (much like the
[KernelType policy](kerneltype.md)).  Below is an example metric class, which
implements the L2 distance:

```c++
class ExampleMetric
{
  // Default constructor is required.
  ExampleMetric() { }

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

Then, this metric can easily be used inside of other mlpack algorithms.  For
example, the code below runs range search on a random dataset with the
`ExampleKernel`, by instantiating a `RangeSearch` object that uses the
`ExampleKernel`.  Then, the number of results are printed.  The `RangeSearch`
class takes three template parameters: `MetricType`, `MatType`, and `TreeType`.
(All three have defaults, so we will just leave `MatType` and `TreeType` to
their defaults.)

```c++
#include <mlpack.hpp>
#include "example_metric.hpp" // A file that contains ExampleKernel.

using namespace mlpack;
using namespace std;

int main()
{
  // Create a random dataset with 10 dimensions and 5000 points.
  arma::mat data = arma::randu<arma::mat>(10, 5000);

  // Instantiate the RangeSearch object with the ExampleKernel.
  RangeSearch<ExampleKernel> rs(data);

  // These vectors will store the results.
  vector<vector<size_t>> neighbors;
  vector<vector<double>> distances;

  // Create a random 10-dimensional query point.
  arma::vec query = arma::randu<arma::vec>(10);

  // Find those points with distance (according to ExampleMetric) between 1 and
  // 2 from the query point.
  rs.Search(query, Range(1.0, 2.0), neighbors, distances);

  // Now, print the number of points inside the desired range.  We know that
  // neighbors and distances will have length 1, since there was only one query
  // point.
  cout << neighbors[0].size() << " points within the range [1.0, 2.0] of the "
      << "query point!" << endl;
}
```

mlpack comes with a number of pre-written metrics that satisfy the `MetricType`
policy:

 - `ManhattanDistance`
 - `EuclideanDistance`
 - `ChebyshevDistance`
 - `MahalanobisDistance`
 - `LMetric` (for arbitrary L-metrics)
 - `IPMetric` (requires a [KernelType](kerneltype.md) parameter)
