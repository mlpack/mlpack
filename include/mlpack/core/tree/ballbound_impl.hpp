/**
 * @file core/tree/ballbound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of BallBound ball bound policy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BALLBOUND_IMPL_HPP
#define MLPACK_CORE_TREE_BALLBOUND_IMPL_HPP

// In case it hasn't been included already.
#include "ballbound.hpp"

#include <string>

namespace mlpack {

//! Empty Constructor.
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>::BallBound() :
    radius(std::numeric_limits<ElemType>::lowest()),
    distance(new DistanceType()),
    ownsDistance(true)
{ /* Nothing to do. */ }

/**
 * Create the ball bound with the specified dimensionality.
 *
 * @param dimension Dimensionality of ball bound.
 */
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>::BallBound(const size_t dimension) :
    radius(std::numeric_limits<ElemType>::lowest()),
    center(dimension),
    distance(new DistanceType()),
    ownsDistance(true)
{ /* Nothing to do. */ }

/**
 * Create the ball bound with the specified radius and center.
 *
 * @param radius Radius of ball bound.
 * @param center Center of ball bound.
 */
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>::BallBound(const ElemType radius,
                                                      const VecType& center) :
    radius(radius),
    center(center),
    distance(new DistanceType()),
    ownsDistance(true)
{ /* Nothing to do. */ }

//! Copy Constructor. To prevent memory leaks.
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>::BallBound(const BallBound& other) :
    radius(other.radius),
    center(other.center),
    distance(other.distance),
    ownsDistance(false)
{ /* Nothing to do. */ }

//! For the same reason as the copy constructor: to prevent memory leaks.
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>&
BallBound<DistanceType, ElemType, VecType>::operator=(
    const BallBound& other)
{
  if (this != &other)
  {
    radius = other.radius;
    center = other.center;
    distance = other.distance;
    ownsDistance = false;
  }
  return *this;
}

//! Move constructor.
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>::BallBound(BallBound&& other) :
    radius(other.radius),
    center(other.center),
    distance(other.distance),
    ownsDistance(other.ownsDistance)
{
  // Fix the other bound.
  other.radius = 0.0;
  other.center = VecType();
  other.distance = NULL;
  other.ownsDistance = false;
}

//! Move assignment operator.
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>&
BallBound<DistanceType, ElemType, VecType>::operator=(
    BallBound&& other)
{
  if (this != &other)
  {
    radius = other.radius;
    center = std::move(other.center);
    distance = other.distance;
    ownsDistance = other.ownsDistance;

    other.radius = 0.0;
    other.center = VecType();
    other.distance = nullptr;
    other.ownsDistance = false;
  }
  return *this;
}

//! Destructor to release allocated memory.
template<typename DistanceType, typename ElemType, typename VecType>
BallBound<DistanceType, ElemType, VecType>::~BallBound()
{
  if (ownsDistance)
    delete distance;
}

//! Get the range in a certain dimension.
template<typename DistanceType, typename ElemType, typename VecType>
RangeType<ElemType>
BallBound<DistanceType, ElemType, VecType>::operator[](const size_t i) const
{
  if (radius < 0)
    return RangeType<ElemType>();
  else
    return RangeType<ElemType>(center[i] - radius, center[i] + radius);
}

/**
 * Determines if a point is within the bound.
 */
template<typename DistanceType, typename ElemType, typename VecType>
bool BallBound<DistanceType, ElemType, VecType>::Contains(const VecType& point)
    const
{
  if (radius < 0)
    return false;
  else
    return distance->Evaluate(center, point) <= radius;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename DistanceType, typename ElemType, typename VecType>
template<typename OtherVecType>
ElemType BallBound<DistanceType, ElemType, VecType>::MinDistance(
    const OtherVecType& point,
    typename std::enable_if_t<IsVector<OtherVecType>::value>* /* junk */) const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return std::max(distance->Evaluate(point, center) - radius, (ElemType) 0.0);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename DistanceType, typename ElemType, typename VecType>
ElemType BallBound<DistanceType, ElemType, VecType>::MinDistance(
    const BallBound& other) const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
  {
    const ElemType delta = distance->Evaluate(center, other.center) - radius -
        other.radius;
    return std::max(delta, (ElemType) 0.0);
  }
}

/**
 * Computes maximum distance.
 */
template<typename DistanceType, typename ElemType, typename VecType>
template<typename OtherVecType>
ElemType BallBound<DistanceType, ElemType, VecType>::MaxDistance(
    const OtherVecType& point,
    typename std::enable_if_t<IsVector<OtherVecType>::value>* /* junk */) const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return distance->Evaluate(point, center) + radius;
}

/**
 * Computes maximum distance.
 */
template<typename DistanceType, typename ElemType, typename VecType>
ElemType BallBound<DistanceType, ElemType, VecType>::MaxDistance(
    const BallBound& other) const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return distance->Evaluate(other.center, center) + radius + other.radius;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename DistanceType, typename ElemType, typename VecType>
template<typename OtherVecType>
RangeType<ElemType> BallBound<DistanceType, ElemType, VecType>::RangeDistance(
    const OtherVecType& point,
    typename std::enable_if_t<IsVector<OtherVecType>::value>* /* junk */) const
{
  if (radius < 0)
    return RangeType<ElemType>(std::numeric_limits<ElemType>::max(),
                               std::numeric_limits<ElemType>::max());
  else
  {
    const ElemType dist = distance->Evaluate(center, point);
    return RangeType<ElemType>(std::max(dist - radius, (ElemType) 0.0),
                               dist + radius);
  }
}

template<typename DistanceType, typename ElemType, typename VecType>
RangeType<ElemType> BallBound<DistanceType, ElemType, VecType>::RangeDistance(
    const BallBound& other) const
{
  if (radius < 0)
    return RangeType<ElemType>(std::numeric_limits<ElemType>::max(),
                               std::numeric_limits<ElemType>::max());
  else
  {
    const ElemType dist = distance->Evaluate(center, other.center);
    const ElemType sumradius = radius + other.radius;
    return RangeType<ElemType>(std::max(dist - sumradius, (ElemType) 0.0),
                               dist + sumradius);
  }
}

/**
 * Expand the bound to include the given point. Algorithm adapted from
 * Jack Ritter, "An Efficient Bounding Sphere" in Graphics Gems (1990).
 * The difference lies in the way we initialize the ball bound. The way we
 * expand the bound is same.
 */
template<typename DistanceType, typename ElemType, typename VecType>
template<typename MatType>
const BallBound<DistanceType, ElemType, VecType>&
BallBound<DistanceType, ElemType, VecType>::operator|=(const MatType& data)
{
  if (radius < 0)
  {
    center = data.col(0);
    radius = 0;
  }

  // Now iteratively add points.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const ElemType dist = distance->Evaluate(center, (VecType) data.col(i));

    // See if the new point lies outside the bound.
    if (dist > radius)
    {
      // Move towards the new point and increase the radius just enough to
      // accommodate the new point.
      const VecType diff = data.col(i) - center;
      center += ((dist - radius) / (2 * dist)) * diff;
      radius = 0.5 * (dist + radius);
    }
  }

  return *this;
}

//! Serialize the BallBound.
template<typename DistanceType, typename ElemType, typename VecType>
template<typename Archive>
void BallBound<DistanceType, ElemType, VecType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(radius));
  ar(CEREAL_NVP(center));

  if (cereal::is_loading<Archive>())
  {
    // If we're loading, delete the local distance metric since we'll have a new
    // one.
    if (ownsDistance)
      delete distance;
  }

  ar(CEREAL_POINTER(distance));
  ar(CEREAL_NVP(ownsDistance));
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP
