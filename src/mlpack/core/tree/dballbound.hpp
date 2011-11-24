/**
 * @file dballbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a ball bound that works in arbitrary metric spaces.
 *
 * @experimental
 */

#ifndef __MLPACK_CORE_TREE_DBALLBOUND_HPP
#define __MLPACK_CORE_TREE_DBALLBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/math/math_misc.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace bound {

/**
 * Ball bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename TMetric = mlpack::metric::SquaredEuclideanDistance,
         typename TPoint = arma::vec>
class DBallBound
{
 public:
  typedef TPoint Point;
  typedef TMetric Metric;

 private:
  double radius_;
  TPoint center_;

 public:
  /**
   * Return the radius of the ball bound.
   */
  double radius() const { return radius_; }

  /**
   * Set the radius of the bound.
   */
  void set_radius(double d) { radius_ = d; }

  /**
   * Return the center point.
   */
  const TPoint& center() const { return center_; }

  /**
   * Return the center point.
   */
  TPoint& center() { return center_; }

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const Point& point) const;

  /**
   * Gets the center.
   *
   * Don't really use this directly.  This is only here for consistency
   * with DHrectBound, so it can plug in more directly if a "centroid"
   * is needed.
   */
  void CalculateMidpoint(Point *centroid) const;

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const Point& point) const;
  double MinDistanceSq(const Point& point) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   */
  double MinDistance(const DBallBound& other) const;
  double MinDistanceSq(const DBallBound& other) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const Point& point) const;
  double MaxDistanceSq(const Point& point) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const DBallBound& other) const;
  double MaxDistanceSq(const DBallBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  math::Range RangeDistance(const DBallBound& other) const;
  math::Range RangeDistanceSq(const DBallBound& other) const;

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
  double MinToMid(const DBallBound& other) const;
  double MinToMidSq(const DBallBound& other) const;

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistance(const DBallBound& other) const;
  double MinimaxDistanceSq(const DBallBound& other) const;

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistance(const DBallBound& other) const;
  double MidDistanceSq(const DBallBound& other) const;
  double MidDistance(const Point& point) const;
};

}; // namespace bound
}; // namespace mlpack

#include "dballbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_DBALLBOUND_HPP
