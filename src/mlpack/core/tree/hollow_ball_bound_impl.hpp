/**
 * @file hollow_ball_bound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of HollowBallBound ball bound metric policy class.
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
    metric(new MetricType()),
    ownsMetric(true)
{ /* Nothing to do. */ }

//! Copy Constructor. To prevent memory leaks.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::HollowBallBound(
    const HollowBallBound& other) :
    radii(other.radii),
    center(other.center),
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
  metric = other.metric;
  ownsMetric = false;

  return *this;
}

//! Move constructor.
template<typename TMetricType, typename ElemType>
HollowBallBound<TMetricType, ElemType>::HollowBallBound(
    HollowBallBound&& other) :
    radii(other.radii),
    center(other.center),
    metric(other.metric),
    ownsMetric(other.ownsMetric)
{
  // Fix the other bound.
  other.radii.Hi() = 0.0;
  other.radii.Lo() = 0.0;
  other.center = arma::Col<ElemType>();
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
    const ElemType dist = metric->Evaluate(center, point);
    return ((dist <= radii.Hi()) && (dist >= radii.Lo()));
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

    bool containOnOneSide = (dist - other.radii.Hi() >= radii.Lo()) &&
        (dist + other.radii.Hi() <= radii.Hi());
    bool containOnEverySide = (dist + radii.Lo() <= other.radii.Lo()) &&
        (dist + other.radii.Hi() <= radii.Hi());

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
    const ElemType dist = metric->Evaluate(point, center);

    const ElemType outerDistance = math::ClampNonNegative(dist - radii.Hi());
    const ElemType innerDistance = math::ClampNonNegative(radii.Lo() - dist);

    return innerDistance + outerDistance;
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
    const ElemType centerDistance = metric->Evaluate(center, other.center);

    const ElemType outerDistance = math::ClampNonNegative(centerDistance -
        radii.Hi() - other.radii.Hi());
    const ElemType innerDistance1 = math::ClampNonNegative(other.radii.Lo() -
        centerDistance - radii.Hi());
    const ElemType innerDistance2 = math::ClampNonNegative(radii.Lo() -
        centerDistance - other.radii.Hi());

    return outerDistance + innerDistance1 + innerDistance2;
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
    const ElemType dist = metric->Evaluate(center, point);
    return math::Range(math::ClampNonNegative(dist - radii.Hi()) +
                       math::ClampNonNegative(radii.Lo() - dist),
                       dist + radii.Hi());
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
    const ElemType dist = metric->Evaluate(center, other.center);
    const ElemType sumradius = radii.Hi() + other.radii.Hi();
    return math::Range(MinDistance(other), dist + sumradius);
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
    radii.Lo() = 0;

    // Now iteratively add points.
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      const ElemType dist = metric->Evaluate(center, data.col(i));

      // See if the new point lies outside the bound.
      if (dist > radii.Hi())
      {
        // Move towards the new point and increase the radius just enough to
        // accommodate the new point.
        const arma::Col<ElemType> diff = data.col(i) - center;
        center += ((dist - radii.Hi()) / (2 * dist)) * diff;
        radii.Hi() = 0.5 * (dist + radii.Hi());
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
      if (dist > radii.Hi())
        radii.Hi() = dist;
      if (dist < radii.Lo())
        radii.Lo() = dist;
    }
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
    radii.Hi() = other.radii.Hi();
    radii.Lo() = other.radii.Lo();
    return *this;
  }

  const ElemType dist = metric->Evaluate(center, other.center);

  if (radii.Hi() < dist + other.radii.Hi())
    radii.Hi() = dist + other.radii.Hi();

  const ElemType innerDist = math::ClampNonNegative(other.radii.Lo() - dist);

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
