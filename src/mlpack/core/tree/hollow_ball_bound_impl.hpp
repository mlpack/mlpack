/**
 * @file hollow_ball_bound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of HollowBallBound ball bound metric policy class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_IMPL_HPP
#define MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_IMPL_HPP

// In case it hasn't been included already.
#include "hollow_ball_bound.hpp"

namespace mlpack {
namespace bound {

//! Empty Constructor.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::HollowBallBound() :
    radii(std::numeric_limits<ElemType>::lowest(),
          std::numeric_limits<ElemType>::lowest()),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

/**
 * Create the hollow ball bound with the specified dimensionality.
 *
 * @param dimension Dimensionality of ball bound.
 */
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::HollowBallBound(const size_t dimension) :
    radii(std::numeric_limits<ElemType>::lowest(),
          std::numeric_limits<ElemType>::lowest()),
    center(dimension),
    hollowCenter(dimension),
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
template<typename TMetricType, typename ElemType>
template<typename VecType>
HollowBallBound<TMetricType, ElemType>::
HollowBallBound(const ElemType innerRadius,
                const ElemType outerRadius,
                const VecType& center) :
    radii(innerRadius,
          outerRadius),
    center(center),
    hollowCenter(center),
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

//! Copy Constructor. To prevent memory leaks.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::HollowBallBound(
    const HollowBallBound& other) :
    radii(other.radii),
    center(other.center),
    hollowCenter(other.hollowCenter),
    metric(other.metric),
    ownsMetric(false)
{ /* Nothing to do. */ }

//! For the same reason as the copy constructor: to prevent memory leaks.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>& HollowBallBound<TMetricType, ElemType>::
operator=(const HollowBallBound& other)
{
  radii = other.radii;
  center = other.center;
  hollowCenter = other.hollowCenter;
  metric = other.metric;
  ownsMetric = false;

  return *this;
}

//! Move constructor.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::HollowBallBound(
    HollowBallBound&& other) :
    radii(other.radii),
    center(std::move(other.center)),
    hollowCenter(std::move(other.hollowCenter)),
    metric(other.metric),
    ownsMetric(other.ownsMetric)
{
  // Fix the other bound.
  other.radii.Hi() = 0.0;
  other.radii.Lo() = 0.0;
  other.center = arma::Col<ElemType>();
  other.hollowCenter = arma::Col<ElemType>();
  other.metric = NULL;
  other.ownsMetric = false;
}

//! Destructor to release allocated memory.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::~HollowBallBound()
{
  if (ownsMetric)
    delete metric;
}

//! Get the range in a certain dimension.
template<typename TMetricType, typename ElemType>
math::RangeType<ElemType> HollowBallBound<TMetricType, ElemType>::operator[](
    const size_t i) const
{
  if (radii.Hi() < 0)
    return math::Range();
  else
    return math::Range(center[i] - radii.Hi(), center[i] + radii.Hi());
}

/**
 * Determines if a point is within the bound.
 */
template<typename TMetricType, typename ElemType>
template<typename VecType>
bool HollowBallBound<TMetricType, ElemType>::Contains(
    const VecType& point) const
{
  if (radii.Hi() < 0)
    return false;
  else
  {
    ElemType dist = metric->Evaluate(center, point);
    if (dist > radii.Hi())
      return false; // The point is situated outside the outer ball.

    // Check if the point is situated outside the hole.
    dist = metric->Evaluate(hollowCenter, point);

    return (dist >= radii.Lo());
  }
}

/**
 * Determines if another bound is within this bound.
 */
template<typename TMetricType, typename ElemType>
bool HollowBallBound<TMetricType, ElemType>::Contains(
    const HollowBallBound& other) const
{
  if (radii.Hi() < 0)
    return false;
  else
  {
    const ElemType dist = metric->Evaluate(center, other.center);
    const ElemType hollowCenterDist = metric->Evaluate(hollowCenter,
        other.center);
    const ElemType hollowHollowDist = metric->Evaluate(hollowCenter,
        other.hollowCenter);

    // The outer ball of the second bound does not contain the hole of the first
    // bound.
    bool containOnOneSide = (hollowCenterDist - other.radii.Hi() >= radii.Lo())
        && (dist + other.radii.Hi() <= radii.Hi());

    // The hole of the second bound contains the hole of the first bound.
    bool containOnEverySide = (hollowHollowDist + radii.Lo() <=
        other.radii.Lo()) && (dist + other.radii.Hi() <= radii.Hi());

    // The first bound has not got a hole.
    bool containAsBall = (radii.Lo() == 0) &&
        (dist + other.radii.Hi() <= radii.Hi());

    return (containOnOneSide || containOnEverySide || containAsBall);
  }
}


/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename TMetricType, typename ElemType>
template<typename VecType>
ElemType HollowBallBound<TMetricType, ElemType>::MinDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType>>* /* junk */) const
{
  if (radii.Hi() < 0)
    return std::numeric_limits<ElemType>::max();
  else
  {
    const ElemType outerDistance = metric->Evaluate(point, center) - radii.Hi();

    if (outerDistance >= 0)
      return outerDistance; // The outer ball does not contain the point.

    // Check if the point is situated in the hole.
    const ElemType innerDistance = math::ClampNonNegative(radii.Lo() -
        metric->Evaluate(point, hollowCenter));

    return innerDistance;
  }
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename TMetricType, typename ElemType>
ElemType HollowBallBound<TMetricType, ElemType>::MinDistance(
    const HollowBallBound& other)
    const
{
  if (radii.Hi() < 0 || other.radii.Hi() < 0)
    return std::numeric_limits<ElemType>::max();
  else
  {
    const ElemType outerDistance = metric->Evaluate(center, other.center) -
        radii.Hi() - other.radii.Hi();
    if (outerDistance >= 0)
      return outerDistance; // The outer hollows do not overlap.

    // Check if the hole of the second bound contains the outer ball of the
    // first bound.
    const ElemType innerDistance1 = other.radii.Lo() -
        metric->Evaluate(center, other.hollowCenter) - radii.Hi();
    if (innerDistance1 >= 0)
      return innerDistance1;

    // Check if the hole of the first bound contains the outer ball of the
    // second bound.
    const ElemType innerDistance2 = math::ClampNonNegative(radii.Lo() -
        metric->Evaluate(hollowCenter, other.center) - other.radii.Hi());

    return innerDistance2;
  }
}

/**
 * Computes maximum distance.
 */
template<typename TMetricType, typename ElemType>
template<typename VecType>
ElemType HollowBallBound<TMetricType, ElemType>::MaxDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
{
  if (radii.Hi() < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return metric->Evaluate(point, center) + radii.Hi();
}

/**
 * Computes maximum distance.
 */
template<typename TMetricType, typename ElemType>
ElemType HollowBallBound<TMetricType, ElemType>::MaxDistance(
  const HollowBallBound& other)
    const
{
  if (radii.Hi() < 0)
    return std::numeric_limits<ElemType>::max();
  else
    return metric->Evaluate(other.center, center) + radii.Hi() +
        other.radii.Hi();
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename TMetricType, typename ElemType>
template<typename VecType>
math::RangeType<ElemType> HollowBallBound<TMetricType, ElemType>::RangeDistance(
    const VecType& point,
    typename boost::enable_if<IsVector<VecType> >* /* junk */) const
{
  if (radii.Hi() < 0)
    return math::Range(std::numeric_limits<ElemType>::max(),
                       std::numeric_limits<ElemType>::max());
  else
  {
    math::RangeType<ElemType> range;
    const ElemType dist = metric->Evaluate(point, center);

    if (dist >= radii.Hi()) // The outer ball does not contain the point.
      range.Lo() = dist - radii.Hi();
    else
    {
      // Check if the point is situated in the hole.
      range.Lo() = math::ClampNonNegative(radii.Lo() -
          metric->Evaluate(point, hollowCenter));
    }
    range.Hi() = dist + radii.Hi();

    return range;
  }
}

template<typename TMetricType, typename ElemType>
math::RangeType<ElemType> HollowBallBound<TMetricType, ElemType>::RangeDistance(
    const HollowBallBound& other) const
{
  if (radii.Hi() < 0)
    return math::Range(std::numeric_limits<ElemType>::max(),
                       std::numeric_limits<ElemType>::max());
  else
  {
    math::RangeType<ElemType> range;

    const ElemType dist = metric->Evaluate(center, other.center);

    const ElemType outerDistance = dist - radii.Hi() - other.radii.Hi();
    if (outerDistance >= 0)
      range.Lo() = outerDistance; // The outer balls do not overlap.
    else
    {
      const ElemType innerDistance1 = other.radii.Lo() -
          metric->Evaluate(center, other.hollowCenter) - radii.Hi();
      // Check if the outer ball of the first bound is contained in the
      // hole of the second bound.
      if (innerDistance1 >= 0)
        range.Lo() = innerDistance1;
      else
      {
        // Check if the outer ball of the second bound is contained in the
        // hole of the first bound.
        range.Lo() = math::ClampNonNegative(radii.Lo() -
            metric->Evaluate(hollowCenter, other.center) - other.radii.Hi());
      }
    }
    range.Hi() = dist + radii.Hi() + other.radii.Hi();
    return range;
  }
}

/**
 * Expand the bound to include the given point. Algorithm adapted from
 * Jack Ritter, "An Efficient Bounding Sphere" in Graphics Gems (1990).
 * The difference lies in the way we initialize the ball bound. The way we
 * expand the bound is same.
 */
template<typename TMetricType, typename ElemType>
template<typename MatType>
const HollowBallBound<TMetricType, ElemType>&
HollowBallBound<TMetricType, ElemType>::operator|=(const MatType& data)
{
  if (radii.Hi() < 0)
  {
    center = data.col(0);
    radii.Hi() = 0;
  }
  if (radii.Lo() < 0)
  {
    hollowCenter = data.col(0);
    radii.Lo() = 0;
  }
  // Now iteratively add points.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const ElemType dist = metric->Evaluate(center, data.col(i));
    const ElemType hollowDist = metric->Evaluate(hollowCenter, data.col(i));

    // See if the new point lies outside the bound.
    if (dist > radii.Hi())
    {
      // Move towards the new point and increase the radius just enough to
      // accommodate the new point.
      const arma::Col<ElemType> diff = data.col(i) - center;
      center += ((dist - radii.Hi()) / (2 * dist)) * diff;
      radii.Hi() = 0.5 * (dist + radii.Hi());
    }
    if (hollowDist < radii.Lo())
      radii.Lo() = hollowDist;
  }

  return *this;
}

/**
 * Expand the bound to include the given bound.
 */
template<typename TMetricType, typename ElemType>
const HollowBallBound<TMetricType, ElemType>&
HollowBallBound<TMetricType, ElemType>::operator|=(const HollowBallBound& other)
{
  if (radii.Hi() < 0)
  {
    center = other.center;
    hollowCenter = other.hollowCenter;
    radii.Hi() = other.radii.Hi();
    radii.Lo() = other.radii.Lo();
    return *this;
  }

  const ElemType dist = metric->Evaluate(center, other.center);
  // Check if the outer balls overlap.
  if (radii.Hi() < dist + other.radii.Hi())
    radii.Hi() = dist + other.radii.Hi();

  const ElemType innerDist = math::ClampNonNegative(other.radii.Lo() -
      metric->Evaluate(hollowCenter, other.hollowCenter));
  // Check if the hole of the first bound is not contained in the hole of the
  // second bound.
  if (radii.Lo() > innerDist)
    radii.Lo() = innerDist;

  return *this;
}


//! Serialize the BallBound.
template<typename TMetricType, typename ElemType>
template<typename Archive>
void HollowBallBound<TMetricType, ElemType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(radii, "radii");
  ar & data::CreateNVP(center, "center");
  ar & data::CreateNVP(hollowCenter, "hollowCenter");

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
