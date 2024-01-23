/**
 * @file core/tree/ballbound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of BallBound ball bound metric policy class.
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
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>::BallBound() :
    radius(std::numeric_limits<ElemType>::lowest()),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

/**
 * Create the ball bound with the specified dimensionality.
 *
 * @param dimension Dimensionality of ball bound.
 */
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>::BallBound(const size_t dimension) :
    radius(std::numeric_limits<ElemType>::lowest()),
    center(dimension),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

/**
 * Create the ball bound with the specified radius and center.
 *
 * @param radius Radius of ball bound.
 * @param center Center of ball bound.
 */
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>::BallBound(const ElemType radius,
                                           const VecType& center) :
    radius(radius),
    center(center),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

//! Copy Constructor. To prevent memory leaks.
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>::BallBound(const BallBound& other) :
    radius(other.radius),
    center(other.center),
    metric(other.metric),
    ownsMetric(false)
{ /* Nothing to do. */ }

//! For the same reason as the copy constructor: to prevent memory leaks.
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>& BallBound<MetricType, VecType>::operator=(
    const BallBound& other)
{
  if (this != &other)
  {
    radius = other.radius;
    center = other.center;
    metric = other.metric;
    ownsMetric = false;
  }
  return *this;
}

//! Move constructor.
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>::BallBound(BallBound&& other) :
    radius(other.radius),
    center(other.center),
    metric(other.metric),
    ownsMetric(other.ownsMetric)
{
  // Fix the other bound.
  other.radius = 0.0;
  other.center = VecType();
  other.metric = NULL;
  other.ownsMetric = false;
}

//! Move assignment operator.
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>& BallBound<MetricType, VecType>::operator=(
    BallBound&& other)
{
  if (this != &other)
  {
    radius = other.radius;
    center = std::move(other.center);
    metric = other.metric;
    ownsMetric = other.ownsMetric;

    other.radius = 0.0;
    other.center = VecType();
    other.metric = nullptr;
    other.ownsMetric = false;
  }
  return *this;
}

//! Destructor to release allocated memory.
template<typename MetricType, typename VecType>
BallBound<MetricType, VecType>::~BallBound()
{
  if (ownsMetric)
    delete metric;
}

//! Get the range in a certain dimension.
template<typename MetricType, typename VecType>
RangeType<typename BallBound<MetricType, VecType>::ElemType>
BallBound<MetricType, VecType>::operator[](const size_t i) const
{
  if (radius < 0)
    return Range();
  else
    return Range(center[i] - radius, center[i] + radius);
}

/**
 * Determines if a point is within the bound.
 */
template<typename MetricType, typename VecType>
bool BallBound<MetricType, VecType>::Contains(const VecType& point) const
{
  if (radius < 0)
    return false;
  else
    return metric->Evaluate(center, point) <= radius;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename MetricType, typename VecType>
template<typename OtherVecType>
typename BallBound<MetricType, VecType>::ElemType
BallBound<MetricType, VecType>::MinDistance(
    const OtherVecType& point,
    typename std::enable_if_t<IsVector<OtherVecType>::value>* /* junk */) const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return std::max(metric->Evaluate(point, center) - radius, (ElemType) 0.0);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename MetricType, typename VecType>
typename BallBound<MetricType, VecType>::ElemType
BallBound<MetricType, VecType>::MinDistance(const BallBound& other)
    const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
  {
    const ElemType delta = metric->Evaluate(center, other.center) - radius -
        other.radius;
    return std::max(delta, (ElemType) 0.0);
  }
}

/**
 * Computes maximum distance.
 */
template<typename MetricType, typename VecType>
template<typename OtherVecType>
typename BallBound<MetricType, VecType>::ElemType
BallBound<MetricType, VecType>::MaxDistance(
    const OtherVecType& point,
    typename std::enable_if_t<IsVector<OtherVecType>::value>* /* junk */) const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return metric->Evaluate(point, center) + radius;
}

/**
 * Computes maximum distance.
 */
template<typename MetricType, typename VecType>
typename BallBound<MetricType, VecType>::ElemType
BallBound<MetricType, VecType>::MaxDistance(const BallBound& other)
    const
{
  if (radius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return metric->Evaluate(other.center, center) + radius + other.radius;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename MetricType, typename VecType>
template<typename OtherVecType>
RangeType<typename BallBound<MetricType, VecType>::ElemType>
BallBound<MetricType, VecType>::RangeDistance(
    const OtherVecType& point,
    typename std::enable_if_t<IsVector<OtherVecType>::value>* /* junk */) const
{
  if (radius < 0)
    return Range(std::numeric_limits<ElemType>::max(),
                       std::numeric_limits<ElemType>::max());
  else
  {
    const ElemType dist = metric->Evaluate(center, point);
    return Range(std::max(dist - radius, (ElemType) 0.0), dist + radius);
  }
}

template<typename MetricType, typename VecType>
RangeType<typename BallBound<MetricType, VecType>::ElemType>
BallBound<MetricType, VecType>::RangeDistance(
    const BallBound& other) const
{
  if (radius < 0)
    return Range(std::numeric_limits<ElemType>::max(),
                       std::numeric_limits<ElemType>::max());
  else
  {
    const ElemType dist = metric->Evaluate(center, other.center);
    const ElemType sumradius = radius + other.radius;
    return Range(std::max(dist - sumradius, (ElemType) 0.0), dist + sumradius);
  }
}

/**
 * Expand the bound to include the given bound.
 *
template<typename MetricType, typename VecType>
const BallBound<VecType>&
BallBound<MetricType, VecType>::operator|=(
    const BallBound<VecType>& other)
{
  double dist = metric->Evaluate(center, other);

  // Now expand the radius as necessary.
  if (dist > radius)
    radius = dist;

  return *this;
}*/

/**
 * Expand the bound to include the given point. Algorithm adapted from
 * Jack Ritter, "An Efficient Bounding Sphere" in Graphics Gems (1990).
 * The difference lies in the way we initialize the ball bound. The way we
 * expand the bound is same.
 */
template<typename MetricType, typename VecType>
template<typename MatType>
const BallBound<MetricType, VecType>&
BallBound<MetricType, VecType>::operator|=(const MatType& data)
{
  if (radius < 0)
  {
    center = data.col(0);
    radius = 0;
  }

  // Now iteratively add points.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const ElemType dist = metric->Evaluate(center, (VecType) data.col(i));

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
template<typename MetricType, typename VecType>
template<typename Archive>
void BallBound<MetricType, VecType>::serialize(
    Archive& ar,
    const uint32_t /* version */)
{
  ar(CEREAL_NVP(radius));
  ar(CEREAL_NVP(center));

  if (cereal::is_loading<Archive>())
  {
    // If we're loading, delete the local metric since we'll have a new one.
    if (ownsMetric)
      delete metric;
  }

  ar(CEREAL_POINTER(metric));
  ar(CEREAL_NVP(ownsMetric));
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP
