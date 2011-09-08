/**
 * @file lmetric.h
 * @author Ryan Curtin
 *
 * Generalized L-metric, allowing both squared distances to be returned as well
 * as non-squared distances.  The squared distances are faster to compute.
 *
 * This also gives several convenience typedefs for commonly used L-metrics.
 */
#ifndef __MLPACK_CORE_KERNELS_LMETRIC_H
#define __MLPACK_CORE_KERNELS_LMETRIC_H

#include <armadillo>

namespace mlpack {
namespace kernel {

/**
 * An L-metric for vector spaces.  The Evaluate() function computes the L-norm
 * of the given power on the two given vectors.  If t_take_root is true, the
 * rooted distance will be returned.  This is slightly faster to compute.
 *
 * @tparam t_pow Power of norm; i.e. t_pow = 1 gives the L1-norm (Manhattan
 *   distance).
 * @tparam t_take_root If true, the t_pow'th root of the result is taken before
 *   it is returned.  In the case of the L2-norm (t_pow = 2), when t_take_root
 *   is true, the squared L2 distance is returned.  It is slightly faster to set
 *   t_take_root = false, because one fewer call to pow() is required.
 */
template<int t_pow, bool t_take_root = false>
class LMetric {
 public:
  /***
   * Default constructor does nothing, but is required to satisfy the Kernel
   * policy.
   */
  LMetric() { }

  /**
   * Computes the distance between two points.
   */
  static double Evaluate(const arma::vec& a, const arma::vec& b);
};

// The implementation is not split into a _impl.h file because it is so simple;
// the unspecialized implementation of the one function is given below.
// Unspecialized implementation.  This should almost never be used...
template<int t_pow, bool t_take_root>
double LMetric<t_pow, t_take_root>::Evaluate(const arma::vec& a,
                                             const arma::vec& b) {
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(fabs(a[i] - b[i]), t_pow);

  if (!t_take_root) // Suboptimal to have this here.
    return sum;

  return pow(sum, (1.0 / t_pow));
}

// Convenience typedefs.

/***
 * The Manhattan (L1) distance.
 */
typedef LMetric<1, false> ManhattanDistance;

/***
 * The squared Euclidean (L2) distance.
 */
typedef LMetric<2, false> SquaredEuclideanDistance;

/***
 * The Euclidean (L2) distance.
 */
typedef LMetric<2, true> EuclideanDistance;

}; // namespace kernel
}; // namespace mlpack

#endif
