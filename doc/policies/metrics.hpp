/*! @page metrics The MetricType policy in mlpack

Many machine learning methods operate with some sort of metric, and often, this
metric can be any arbitrary metric.  For instance, consider the problem of
nearest neighbor search; one can find the nearest neighbor of a point with
respect to the standard Euclidean distance, or the Manhattan (city-block)
distance.  The actual search techniques, though, remain the same.  And this is
true of many machine learning methods: the specific metric that is used can be
any valid metric.

mlpack algorithms, when possible, allow the use of an arbitrary metric via the
use of the \c MetricType template parameter.  Any metric passed as a
\c MetricType template parameter will need to have

 - an \c Evaluate function
 - a default constructor.

The signature of the \c Evaluate function is straightforward:

@code
template<typename VecTypeA, typename VecTypeB>
double Evaluate(const VecTypeA& a, const VecTypeB& b);
@endcode

The function takes two vector arguments, \c a and \c b, and returns a \c double
that is the evaluation of the metric between the two arguments.  So, for a
particular metric \f$d(\cdot, \cdot)\f$, the \c Evaluate() function should
return \f$d(a, b)\f$.

The arguments \c a and \c b, of types \c VecTypeA and \c VecTypeB, respectively,
will be an Armadillo-like vector type (usually \c arma::vec, \c arma::sp_vec, or
similar).  In general it should be valid to assume that \c VecTypeA is a class
with the same API as \c arma::vec.

Note that for metrics that do not hold any state, the \c Evaluate() method can
be marked as \c static.

Overall, the \c MetricType template policy is quite simple (much like the
\ref kernels KernelType policy).  Below is an example metric class, which
implements the L2 distance:

@code
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
@endcode

Then, this metric can easily be used inside of other mlpack algorithms.  For
example, the code below runs range search on a random dataset with the
\c ExampleKernel, by instantiating a \c mlpack::range::RangeSearch object that
uses the \c ExampleKernel.  Then, the number of results are printed.  The \c
RangeSearch class takes three template parameters: \c MetricType, \c MatType,
and \c TreeType.  (All three have defaults, so we will just leave \c MatType and
\c TreeType to their defaults.)

@code
#include <mlpack/core.hpp>
#include <mlpack/methods/range_search/range_search.hpp>
#include "example_metric.hpp" // A file that contains ExampleKernel.

using namespace mlpack;
using namespace mlpack::range;
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
  rs.Search(query, math::Range(1.0, 2.0), neighbors, distances);

  // Now, print the number of points inside the desired range.  We know that
  // neighbors and distances will have length 1, since there was only one query
  // point.
  cout << neighbors[0].size() << " points within the range [1.0, 2.0] of the "
      << "query point!" << endl;
}
@endcode

mlpack comes with a number of pre-written metrics that satisfy the \c MetricType
policy:

 - mlpack::metric::ManhattanDistance
 - mlpack::metric::EuclideanDistance
 - mlpack::metric::ChebyshevDistance
 - mlpack::metric::MahalanobisDistance
 - mlpack::metric::LMetric (for arbitrary L-metrics)
 - mlpack::metric::IPMetric (requires a \ref kernels KernelType parameter)

*/
