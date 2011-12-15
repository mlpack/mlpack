/**
 * @file ballbound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of BallBound ball bound metric policy class.
 *
 * @experimental
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
  return math::Range(center[i] - radius, center[i] + radius);
}

/**
 * Determines if a point is within the bound.
 */
template<typename VecType>
bool BallBound<VecType>::Contains(const VecType& point) const
{
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
  return math::ClampNonNegative(metric::EuclideanDistance::Evaluate(point,
      center) - radius);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename VecType>
double BallBound<VecType>::MinDistance(const BallBound& other) const
{
  double delta = metric::EuclideanDistance::Evaluate(center, other.center)
      - radius - other.radius;
  return math::ClampNonNegative(delta);
}

/**
 * Computes maximum distance.
 */
template<typename VecType>
double BallBound<VecType>::MaxDistance(const VecType& point) const
{
  return metric::EuclideanDistance::Evaluate(point, center) + radius;
}

/**
 * Computes maximum distance.
 */
template<typename VecType>
double BallBound<VecType>::MaxDistance(const BallBound& other) const
{
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
  double dist = metric::EuclideanDistance::Evaluate(center, point);
  return math::Range(math::ClampNonNegative(dist - radius),
                                            dist + radius);
}

template<typename VecType>
math::Range BallBound<VecType>::RangeDistance(
    const BallBound& other) const
{
  double dist = metric::EuclideanDistance::Evaluate(center, other.center);
  double sumradius = radius + other.radius;
  return math::Range(math::ClampNonNegative(dist - sumradius),
                                            dist + sumradius);
}

/**
 * Expand the bound to include the given bound.
 */
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
}

/**
 * Expand the bound to include the given point.
 */
template<typename VecType>
const BallBound<VecType>&
BallBound<VecType>::operator|=(const VecType& point)
{
  double dist = metric::EuclideanDistance::Evaluate(center, point);

  if (dist > radius)
    radius = dist;

  return *this;
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP
