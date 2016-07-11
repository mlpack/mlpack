/**
 * @file hollow_ball_bound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of HollowBallBound ball bound metric policy class.
 *
 * @experimental
 */
#ifndef MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_IMPL_HPP
#define MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_IMPL_HPP

// In case it hasn't been included already.
#include "hollow_ball_bound.hpp"

#include <string>

namespace mlpack {
namespace bound {

//! Empty Constructor.
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>::HollowBallBound() :
    innerRadius(std::numeric_limits<ElemType>::lowest()),
    outerRadius(std::numeric_limits<ElemType>::lowest()),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

/**
 * Create the hollow ball bound with the specified dimensionality.
 *
 * @param dimension Dimensionality of ball bound.
 */
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>::HollowBallBound(const size_t dimension) :
    innerRadius(std::numeric_limits<ElemType>::lowest()),
    outerRadius(std::numeric_limits<ElemType>::lowest()),
    center(dimension),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

/**
 * Create the hollow ball bound with the specified radii and center.
 *
 * @param innerRadius Inner radius of hollow ball bound.
 * @param outerRadius Outer radius of hollow ball bound.
 * @param center Center of hollow ball bound.
 */
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>::
HollowBallBound(const ElemType innerRadius,
                const ElemType outerRadius,
                const VecType& center) :
    innerRadius(innerRadius),
    outerRadius(outerRadius),
    center(center),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

//! Copy Constructor. To prevent memory leaks.
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>::HollowBallBound(
    const HollowBallBound& other) :
    innerRadius(other.innerRadius),
    outerRadius(other.outerRadius),
    center(other.center),
    metric(other.metric),
    ownsMetric(false)
{ /* Nothing to do. */ }

//! For the same reason as the copy constructor: to prevent memory leaks.
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>& HollowBallBound<MetricType, VecType>::
operator=(const HollowBallBound& other)
{
  innerRadius = other.innerRadius;
  outerRadius = other.outerRadius;
  center = other.center;
  metric = other.metric;
  ownsMetric = false;

  return *this;
}

//! Move constructor.
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>::HollowBallBound(HollowBallBound&& other) :
    innerRadius(other.innerRadius),
    outerRadius(other.outerRadius),
    center(other.center),
    metric(other.metric),
    ownsMetric(other.ownsMetric)
{
  // Fix the other bound.
  other.innerRadius = 0.0;
  other.outerRadius = 0.0;
  other.center = VecType();
  other.metric = NULL;
  other.ownsMetric = false;
}

//! Destructor to release allocated memory.
template<typename MetricType, typename VecType>
HollowBallBound<MetricType, VecType>::~HollowBallBound()
{
  if (ownsMetric)
    delete metric;
}

//! Get the range in a certain dimension.
template<typename MetricType, typename VecType>
math::RangeType<typename HollowBallBound<MetricType, VecType>::ElemType>
HollowBallBound<MetricType, VecType>::operator[](const size_t i) const
{
  if (outerRadius < 0)
    return math::Range();
  else
    return math::Range(center[i] - outerRadius, center[i] + outerRadius);
}

/**
 * Determines if a point is within the bound.
 */
template<typename MetricType, typename VecType>
bool HollowBallBound<MetricType, VecType>::Contains(const VecType& point) const
{
  if (outerRadius < 0)
    return false;
  else
  {
    const ElemType dist = metric->Evaluate(center, point);
    return ((dist <= outerRadius) && (dist >= innerRadius));
  }
}

/**
 * Determines if another bound is within this bound.
 */
template<typename MetricType, typename VecType>
bool HollowBallBound<MetricType, VecType>::Contains(
    const HollowBallBound& other) const
{
  if (outerRadius < 0)
    return false;
  else
  {
    const ElemType dist = metric->Evaluate(center, other.center);

    bool containOnOneSide = (dist - other.outerRadius >= innerRadius) &&
        (dist + other.outerRadius <= outerRadius);
    bool containOnEverySide = (dist + innerRadius <= other.innerRadius) &&
        (dist + other.outerRadius <= outerRadius);

    bool containAsBall = (innerRadius == 0) &&
        (dist + other.outerRadius <= outerRadius);

    return (containOnOneSide || containOnEverySide || containAsBall);
  }
}


/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename MetricType, typename VecType>
template<typename OtherVecType>
typename HollowBallBound<MetricType, VecType>::ElemType
HollowBallBound<MetricType, VecType>::MinDistance(
    const OtherVecType& point,
    typename boost::enable_if<IsVector<OtherVecType>>* /* junk */) const
{
  if (outerRadius < 0)
    return std::numeric_limits<ElemType>::max();
  else
  {
    const ElemType dist = metric->Evaluate(point, center);

    const ElemType outerDistance = math::ClampNonNegative(dist - outerRadius);
    const ElemType innerDistance = math::ClampNonNegative(innerRadius - dist);

    return innerDistance + outerDistance;
  }
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename MetricType, typename VecType>
typename HollowBallBound<MetricType, VecType>::ElemType
HollowBallBound<MetricType, VecType>::MinDistance(const HollowBallBound& other)
    const
{
  if (outerRadius < 0 || other.outerRadius < 0)
    return std::numeric_limits<ElemType>::max();
  else
  {
    const ElemType centerDistance = metric->Evaluate(center, other.center);

    const ElemType outerDistance = math::ClampNonNegative(centerDistance -
        outerRadius - other.outerRadius);
    const ElemType innerDistance1 = math::ClampNonNegative(other.innerRadius -
        centerDistance - outerRadius);
    const ElemType innerDistance2 = math::ClampNonNegative(innerRadius -
        centerDistance - other.outerRadius);

    return outerDistance + innerDistance1 + innerDistance2;
  }
}

/**
 * Computes maximum distance.
 */
template<typename MetricType, typename VecType>
template<typename OtherVecType>
typename HollowBallBound<MetricType, VecType>::ElemType
HollowBallBound<MetricType, VecType>::MaxDistance(
    const OtherVecType& point,
    typename boost::enable_if<IsVector<OtherVecType> >* /* junk */) const
{
  if (outerRadius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return metric->Evaluate(point, center) + outerRadius;
}

/**
 * Computes maximum distance.
 */
template<typename MetricType, typename VecType>
typename HollowBallBound<MetricType, VecType>::ElemType
HollowBallBound<MetricType, VecType>::MaxDistance(const HollowBallBound& other)
    const
{
  if (outerRadius < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return metric->Evaluate(other.center, center) + outerRadius +
        other.outerRadius;
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename MetricType, typename VecType>
template<typename OtherVecType>
math::RangeType<typename HollowBallBound<MetricType, VecType>::ElemType>
HollowBallBound<MetricType, VecType>::RangeDistance(
    const OtherVecType& point,
    typename boost::enable_if<IsVector<OtherVecType> >* /* junk */) const
{
  if (outerRadius < 0)
    return math::Range(std::numeric_limits<ElemType>::max(),
                       std::numeric_limits<ElemType>::max());
  else
  {
    const ElemType dist = metric->Evaluate(center, point);
    return math::Range(math::ClampNonNegative(dist - outerRadius) +
                       math::ClampNonNegative(innerRadius - dist),
                       dist + outerRadius);
  }
}

template<typename MetricType, typename VecType>
math::RangeType<typename HollowBallBound<MetricType, VecType>::ElemType>
HollowBallBound<MetricType, VecType>::RangeDistance(
    const HollowBallBound& other) const
{
  if (outerRadius < 0)
    return math::Range(std::numeric_limits<ElemType>::max(),
                       std::numeric_limits<ElemType>::max());
  else
  {
    const ElemType dist = metric->Evaluate(center, other.center);
    const ElemType sumradius = outerRadius + other.outerRadius;
    return math::Range(MinDistance(other), dist + sumradius);
  }
}

/**
 * Expand the bound to include the given point. Algorithm adapted from
 * Jack Ritter, "An Efficient Bounding Sphere" in Graphics Gems (1990).
 * The difference lies in the way we initialize the ball bound. The way we
 * expand the bound is same.
 */
template<typename MetricType, typename VecType>
template<typename MatType>
const HollowBallBound<MetricType, VecType>&
HollowBallBound<MetricType, VecType>::operator|=(const MatType& data)
{
  if (outerRadius < 0)
  {
    center = data.col(0);
    outerRadius = 0;
    innerRadius = 0;

    // Now iteratively add points.
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      const ElemType dist = metric->Evaluate(center, (VecType) data.col(i));

      // See if the new point lies outside the bound.
      if (dist > outerRadius)
      {
        // Move towards the new point and increase the radius just enough to
        // accommodate the new point.
        const VecType diff = data.col(i) - center;
        center += ((dist - outerRadius) / (2 * dist)) * diff;
        outerRadius = 0.5 * (dist + outerRadius);
      }
    }
  }
  else
  {
    // Now iteratively add points.
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      const ElemType dist = metric->Evaluate(center, data.col(i));

      // See if the new point lies outside the bound.
      if (dist > outerRadius)
        outerRadius = dist;
      if (dist < innerRadius)
        innerRadius = dist;
    }
  }

  return *this;
}

/**
 * Expand the bound to include the given bound.
 */
template<typename MetricType, typename VecType>
const HollowBallBound<MetricType, VecType>&
HollowBallBound<MetricType, VecType>::operator|=(const HollowBallBound& other)
{
  if (outerRadius < 0)
  {
    center = other.center;
    outerRadius = other.outerRadius;
    innerRadius = other.innerRadius;
    return *this;
  }

  const ElemType dist = metric->Evaluate(center, other.center);

  if (outerRadius < dist + other.outerRadius)
    outerRadius = dist + other.outerRadius;

  const ElemType innerDist = math::ClampNonNegative(other.innerRadius - dist);

  if (innerRadius > innerDist)
    innerRadius = innerDist;

  return *this;
}


//! Serialize the BallBound.
template<typename MetricType, typename VecType>
template<typename Archive>
void HollowBallBound<MetricType, VecType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(innerRadius, "innerRadius");
  ar & data::CreateNVP(outerRadius, "outerRadius");
  ar & data::CreateNVP(center, "center");

  if (Archive::is_loading::value)
  {
    // If we're loading, delete the local metric since we'll have a new one.
    if (ownsMetric)
      delete metric;
  }

  ar & data::CreateNVP(metric, "metric");
  ar & data::CreateNVP(ownsMetric, "ownsMetric");
}

} // namespace bound
} // namespace mlpack

#endif // MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_IMPL_HPP
