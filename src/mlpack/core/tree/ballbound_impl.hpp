/**
 * @file ballbound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of BallBound ball bound metric policy class.
 *
 * @experimental
 *
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
#ifndef __MLPACK_CORE_TREE_BALLBOUND_IMPL_HPP
#define __MLPACK_CORE_TREE_BALLBOUND_IMPL_HPP

// In case it hasn't been included already.
#include "ballbound.hpp"

namespace mlpack {
namespace bound {

//! Get the range in a certain dimension.
template<typename VecType>
math::Range BallBound<VecType>::operator[](const size_t i) const
{
  if (radius < 0)
    return math::Range();
  else
    return math::Range(center[i] - radius, center[i] + radius);
}

/**
 * Determines if a point is within the bound.
 */
template<typename VecType>
bool BallBound<VecType>::Contains(const VecType& point) const
{
  if (radius < 0)
    return false;
  else
    return metric::EuclideanDistance::Evaluate(center, point) <= radius;
}

/**
 * Gets the center.
 *
 * Don't really use this directly.  This is only here for consistency
 * with DHrectBound, so it can plug in more directly if a "centroid"
 * is needed.
 */
template<typename VecType>
void BallBound<VecType>::CalculateMidpoint(VecType& centroid) const
{
  centroid = center;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename VecType>
double BallBound<VecType>::MinDistance(const VecType& point) const
{
  if (radius < 0)
    return DBL_MAX;
  else
    return math::ClampNonNegative(metric::EuclideanDistance::Evaluate(point,
        center) - radius);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename VecType>
double BallBound<VecType>::MinDistance(const BallBound& other) const
{
  if (radius < 0)
    return DBL_MAX;
  else
  {
    double delta = metric::EuclideanDistance::Evaluate(center, other.center)
        - radius - other.radius;
    return math::ClampNonNegative(delta);
  }
}

/**
 * Computes maximum distance.
 */
template<typename VecType>
double BallBound<VecType>::MaxDistance(const VecType& point) const
{
  if (radius < 0)
    return DBL_MAX;
  else
    return metric::EuclideanDistance::Evaluate(point, center) + radius;
}

/**
 * Computes maximum distance.
 */
template<typename VecType>
double BallBound<VecType>::MaxDistance(const BallBound& other) const
{
  if (radius < 0)
    return DBL_MAX;
  else
    return metric::EuclideanDistance::Evaluate(other.center, center) + radius
        + other.radius;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename VecType>
math::Range BallBound<VecType>::RangeDistance(const VecType& point)
    const
{
  if (radius < 0)
    return math::Range(DBL_MAX, DBL_MAX);
  else
  {
    double dist = metric::EuclideanDistance::Evaluate(center, point);
    return math::Range(math::ClampNonNegative(dist - radius),
                                              dist + radius);
  }
}

template<typename VecType>
math::Range BallBound<VecType>::RangeDistance(
    const BallBound& other) const
{
  if (radius < 0)
    return math::Range(DBL_MAX, DBL_MAX);
  else
  {
    double dist = metric::EuclideanDistance::Evaluate(center, other.center);
    double sumradius = radius + other.radius;
    return math::Range(math::ClampNonNegative(dist - sumradius),
                                              dist + sumradius);
  }
}

/**
 * Expand the bound to include the given bound.
 *
template<typename VecType>
const BallBound<VecType>&
BallBound<VecType>::operator|=(
    const BallBound<VecType>& other)
{
  double dist = metric::EuclideanDistance::Evaluate(center, other);

  // Now expand the radius as necessary.
  if (dist > radius)
    radius = dist;

  return *this;
}*/

/**
 * Expand the bound to include the given point.
 */
template<typename VecType>
template<typename MatType>
const BallBound<VecType>&
BallBound<VecType>::operator|=(const MatType& data)
{
  if (radius < 0)
  {
    center = data.col(0);
    radius = 0;
  }

  // Now iteratively add points.  There is probably a closed-form solution to
  // find the minimum bounding circle, and it is probably faster.
  for (size_t i = 1; i < data.n_cols; ++i)
  {
    double dist = metric::EuclideanDistance::Evaluate(center, (VecType)
        data.col(i)) - radius;

    if (dist > 0)
    {
      // Move (dist / 2) towards the new point and increase radius by
      // (dist / 2).
      arma::vec diff = data.col(i) - center;
      center += 0.5 * diff;
      radius += 0.5 * dist;
    }
  }

  return *this;
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP
