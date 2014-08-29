/**
 * @file lmetric.hpp
 * @author Ryan Curtin
 *
 * Generalized L-metric, allowing both squared distances to be returned as well
 * as non-squared distances.  The squared distances are faster to compute.
 *
 * This also gives several convenience typedefs for commonly used L-metrics.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_METRICS_LMETRIC_HPP
#define __MLPACK_CORE_METRICS_LMETRIC_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace metric {

/**
 * The L_p metric for arbitrary integer p, with an option to take the root.
 *
 * This class implements the standard L_p metric for two arbitrary vectors @f$ x
 * @f$ and @f$ y @f$ of dimensionality @f$ n @f$:
 *
 * @f[
 * d(x, y) = \left( \sum_{i = 1}^{n} | x_i - y_i |^p \right)^{\frac{1}{p}}.
 * @f]
 *
 * The value of p is given as a template parameter.
 *
 * In addition, the function @f$ d(x, y) @f$ can be simplified, neglecting the
 * p-root calculation.  This is done by specifying the TakeRoot template
 * parameter to be false.  Then,
 *
 * @f[
 * d(x, y) = \sum_{i = 1}^{n} | x_i - y_i |^p
 * @f]
 *
 * It is faster to compute that distance, so TakeRoot is by default off.
 * However, when TakeRoot is false, the distance given is not actually a true
 * metric -- it does not satisfy the triangle inequality.  Some MLPACK methods
 * do not require the triangle inequality to operate correctly (such as the
 * BinarySpaceTree), but setting TakeRoot = false in some cases will cause
 * incorrect results.
 *
 * A few convenience typedefs are given:
 *
 *  - ManhattanDistance
 *  - EuclideanDistance
 *  - SquaredEuclideanDistance
 *
 * @tparam Power Power of metric; i.e. Power = 1 gives the L1-norm (Manhattan
 *    distance).
 * @tparam TakeRoot If true, the Power'th root of the result is taken before it
 *    is returned.  Setting this to false causes the metric to not satisfy the
 *    Triangle Inequality (be careful!).
 */
template<int Power, bool TakeRoot = true>
class LMetric
{
 public:
  /***
   * Default constructor does nothing, but is required to satisfy the Kernel
   * policy.
   */
  LMetric() { }

  /**
   * Computes the distance between two points.
   */
  template<typename VecType1, typename VecType2>
  static double Evaluate(const VecType1& a, const VecType2& b);
  std::string ToString() const;
};

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

/***
 * The L-infinity distance
 */
typedef LMetric<INT_MAX, false> ChebyshevDistance;


}; // namespace metric
}; // namespace mlpack

// Include implementation.
#include "lmetric_impl.hpp"

#endif
