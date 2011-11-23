/**
 * @file dballbound_impl.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Implementation of DBallBound ball bound metric policy class.
 *
 * @experimental
 */
#ifndef __MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP
#define __MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP

#include <mlpack/core.h>

namespace mlpack {
namespace bound {

/**
 * Determines if a point is within the bound.
 */
template<typename TMetric, typename TPoint>
bool DBallBound<TMetric, TPoint>::Contains(const Point& point) const
{
  return MidDistance(point) <= radius_;
}

/**
 * Gets the center.
 *
 * Don't really use this directly.  This is only here for consistency
 * with DHrectBound, so it can plug in more directly if a "centroid"
 * is needed.
 */
template<typename TMetric, typename TPoint>
void DBallBound<TMetric, TPoint>::CalculateMidpoint(Point *centroid) const
{
  (*centroid) = center_;
}

/**
 * Calculates minimum bound-to-point squared distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistance(const Point& point) const
{
  return math::ClampNonNegative(MidDistance(point) - radius_);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistanceSq(const Point& point) const
{
  return std::pow(MinDistance(point), 2);
}

/**
 * Calculates minimum bound-to-bound squared distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistance(const DBallBound& other) const
{
  double delta = MidDistance(other.center_) - radius_ - other.radius_;
  return math::ClampNonNegative(delta);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinDistanceSq(const DBallBound& other) const
{
  return std::pow(MinDistance(other), 2);
}

/**
 * Computes maximum distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistance(const Point& point) const
{
  return MidDistance(point) + radius_;
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistanceSq(const Point& point) const
{
  return std::pow(MaxDistance(point), 2);
}

/**
 * Computes maximum distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistance(const DBallBound& other) const
{
  return MidDistance(other.center_) + radius_ + other.radius_;
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MaxDistanceSq(const DBallBound& other) const
{
  return std::pow(MaxDistance(other), 2);
}

/**
 * Calculates minimum and maximum bound-to-bound squared distance.
 *
 * Example: bound1.MinDistanceSq(other) for minimum squared distance.
 */
template<typename TMetric, typename TPoint>
math::Range DBallBound<TMetric, TPoint>::RangeDistance(
    const DBallBound& other) const
{
  double delta = MidDistance(other.center_);
  double sumradius = radius_ + other.radius_;
  return math::Range(
      math::ClampNonNegative(delta - sumradius),
      delta + sumradius);
}

template<typename TMetric, typename TPoint>
math::Range DBallBound<TMetric, TPoint>::RangeDistanceSq(
    const DBallBound& other) const
{
  double delta = MidDistance(other.center_);
  double sumradius = radius_ + other.radius_;
  return math::Range(
      std::pow(math::ClampNonNegative(delta - sumradius), 2),
      std::pow(delta + sumradius, 2));
}

/**
 * Calculates closest-to-their-midpoint bounding box distance,
 * i.e. calculates their midpoint and finds the minimum box-to-point
 * distance.
 *
 * Equivalent to:
 * <code>
 * other.CalcMidpoint(&other_midpoint)
 * return MinDistanceSqToPoint(other_midpoint)
 * </code>
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinToMid(const DBallBound& other) const
{
  double delta = MidDistance(other.center_) - radius_;
  return math::ClampNonNegative(delta);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinToMidSq(const DBallBound& other) const
{
  return std::pow(MinToMid(other), 2);
}

/**
 * Computes minimax distance, where the other node is trying to avoid me.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinimaxDistance(
    const DBallBound& other) const
{
  double delta = MidDistance(other.center_) + other.radius_ - radius_;
  return math::ClampNonNegative(delta);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MinimaxDistanceSq(
    const DBallBound& other) const
{
  return std::pow(MinimaxDistance(other), 2);
}

/**
 * Calculates midpoint-to-midpoint bounding box distance.
 */
template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MidDistance(const DBallBound& other) const
{
  return MidDistance(other.center_);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MidDistanceSq(const DBallBound& other) const
{
  return std::pow(MidDistance(other), 2);
}

template<typename TMetric, typename TPoint>
double DBallBound<TMetric, TPoint>::MidDistance(const Point& point) const
{
  return Metric::Evaluate(center_, point);
}

}; // namespace bound
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_DBALLBOUND_IMPL_HPP
