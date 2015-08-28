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
\c MetricType template parameter will need to implement an \c Evaluate function
and a default constructor.

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

mlpack comes with a number of pre-written metrics that satisfy the \c MetricType
policy:

 - mlpack::metric::ManhattanDistance
 - mlpack::metric::EuclideanDistance
 - mlpack::metric::ChebyshevDistance
 - mlpack::metric::MahalanobisDistance
 - mlpack::metric::LMetric (for arbitrary L-metrics)
 - mlpack::metric::IPMetric (requires a \ref kernels KernelType parameter)

*/
