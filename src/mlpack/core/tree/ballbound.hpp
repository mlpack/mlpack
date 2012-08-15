/**
 * @file ballbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a ball bound that works in arbitrary metric spaces.
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

#ifndef __MLPACK_CORE_TREE_BALLBOUND_HPP
#define __MLPACK_CORE_TREE_BALLBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace bound {

/**
 * Ball bound that works in the regular Euclidean metric space.
 *
 * @tparam VecType Type of vector (arma::vec or arma::spvec).
 */
template<typename VecType = arma::vec>
class BallBound
{
 public:
  typedef VecType Vec;

 private:
  double radius;
  VecType center;

 public:
  BallBound() : radius(0) { }

  /**
   * Create the ball bound with the specified dimensionality.
   *
   * @param dimension Dimensionality of ball bound.
   */
  BallBound(const size_t dimension) : radius(0), center(dimension) { }

  /**
   * Create the ball bound with the specified radius and center.
   *
   * @param radius Radius of ball bound.
   * @param center Center of ball bound.
   */
  BallBound(const double radius, const VecType& center) :
      radius(radius), center(center) { }

  //! Get the radius of the ball.
  double Radius() const { return radius; }
  //! Modify the radius of the ball.
  double& Radius() { return radius; }

  //! Get the center point of the ball.
  const VecType& Center() const { return center; }
  //! Modify the center point of the ball.
  VecType& Center() { return center; }

  // Get the range in a certain dimension.
  math::Range operator[](const size_t i) const;

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const VecType& point) const;

  /**
   * Gets the center.
   *
   * Don't really use this directly.  This is only here for consistency
   * with DHrectBound, so it can plug in more directly if a "centroid"
   * is needed.
   */
  void CalculateMidpoint(VecType& centroid) const;

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const VecType& point) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   */
  double MinDistance(const BallBound& other) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const VecType& point) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const BallBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   */
  math::Range RangeDistance(const VecType& other) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum distance.
   */
  math::Range RangeDistance(const BallBound& other) const;

  /**
   * Expand the bound to include the given node.
   */
  const BallBound& operator|=(const BallBound& other);

  /**
   * Expand the bound to include the given point.  The centroid is recalculated
   * to be the center of all of the given points.
   *
   * @tparam MatType Type of matrix; could be arma::mat, arma::spmat, or a
   *     vector.
   * @tparam data Data points to add.
   */
  template<typename MatType>
  const BallBound& operator|=(const MatType& data);
};

}; // namespace bound
}; // namespace mlpack

#include "ballbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_DBALLBOUND_HPP
