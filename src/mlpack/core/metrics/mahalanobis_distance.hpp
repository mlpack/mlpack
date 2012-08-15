/***
 * @file mahalanobis_dstance.h
 * @author Ryan Curtin
 *
 * The Mahalanobis distance.
 * This file is part of MLPACK 1.0.2.
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
#ifndef __MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_HPP
#define __MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace metric {

/**
 * The Mahalanobis distance, which is essentially a stretched Euclidean
 * distance.  Given a square covariance matrix @f$ Q @f$ of size @f$ d @f$ x
 * @f$ d @f$, where @f$ d @f$ is the dimensionality of the points it will be
 * evaluating, and given two vectors @f$ x @f$ and @f$ y @f$ also of
 * dimensionality @f$ d @f$,
 *
 * @f[
 * d(x, y) = \sqrt{(x - y)^T Q (x - y)}
 * @f]
 *
 * where Q is the covariance matrix.
 *
 * Because each evaluation multiplies (x_1 - x_2) by the covariance matrix, it
 * may be much quicker to use an LMetric and simply stretch the actual dataset
 * itself before performing any evaluations.  However, this class is provided
 * for convenience.
 *
 * Similar to the LMetric class, this offers a template parameter t_take_root
 * which, when set to false, will instead evaluate the distance
 *
 * @f[
 * d(x, y) = (x - y)^T Q (x - y)
 * @f]
 *
 * which is faster to evaluate.
 *
 * @tparam t_take_root If true, takes the root of the output.  It is slightly
 *   faster to leave this at the default of false.
 */
template<bool t_take_root = false>
class MahalanobisDistance
{
 public:
  /**
   * Initialize the Mahalanobis distance with the empty matrix as covariance.
   * Because we don't actually know the size of the vectors we will be using, we
   * delay creation of the covariance matrix until evaluation.
   */
  MahalanobisDistance() : covariance(0, 0) { }

  /**
   * Initialize the Mahalanobis distance with the given covariance matrix.  The
   * given covariance matrix will be copied (this is not optimal).
   *
   * @param covariance The covariance matrix to use for this distance.
   */
  MahalanobisDistance(const arma::mat& covariance) : covariance(covariance) { }

  /**
   * Evaluate the distance between the two given points using this Mahalanobis
   * distance.
   *
   * @param a First vector.
   * @param b Second vector.
   */
  template<typename VecType1, typename VecType2>
  double Evaluate(const VecType1& a, const VecType2& b);

  /**
   * Access the covariance matrix.
   *
   * @return Constant reference to the covariance matrix.
   */
  const arma::mat& Covariance() const { return covariance; }

  /**
   * Modify the covariance matrix.
   *
   * @return Reference to the covariance matrix.
   */
  arma::mat& Covariance() { return covariance; }

 private:
  //! The covariance matrix associated with this distance.
  arma::mat covariance;
};

}; // namespace distance
}; // namespace mlpack

#include "mahalanobis_distance_impl.hpp"

#endif
