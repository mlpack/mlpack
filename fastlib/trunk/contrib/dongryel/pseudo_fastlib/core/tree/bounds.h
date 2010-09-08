/**
 * @file tree/bounds.h
 *
 * Bounds that are useful for binary space partitioning trees.
 */

#ifndef CORE_TREE_BOUNDS_H
#define CORE_TREE_BOUNDS_H

#include "core/math/math_lib.h"
#include "core/math/range.h"

namespace core {
namespace tree {

/**
 * Ball bound that works in arbitrary metric spaces.
 */
template<typename TMetric, typename TPoint>
class BallBound {
  public:
    typedef TPoint Point;
    typedef TMetric MetricType;

  private:
    double radius_;
    TPoint center_;

  public:
    double radius() const {
      return radius_;
    }

    void set_radius(double d) {
      radius_ = d;
    }

    const TPoint& center() const {
      return center_;
    }

    TPoint& center() {
      return center_;
    }

    /**
     * Determines if a point is within this bound.
     */
    bool Contains(const Point& point) const {
      return MidDistance(point) <= radius_;
    }

    /**
     * Gets the center.
     *
     * Don't really use this directly.  This is only here for consistency
     * with DHrectBound, so it can plug in more directly if a "centroid"
     * is needed.
     */
    void CalculateMidpoint(Point *centroid) const {
      center_(*centroid);
    }

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistance(
      const MetricType &metric, const Point& point) const {

      return std::max(MidDistance(metric, point) - radius_, 0.0);
    }

    double MinDistanceSq(
      const MetricType &metric, const Point& point) const {

      return core::math::Pow<2, 1>(MinDistance(metric, point));
    }

    /**
     * Calculates minimum bound-to-bound squared distance.
     */
    double MinDistance(const MetricType &metric, const BallBound& other) const {
      double delta =
        MidDistance(metric, other.center_) - radius_ - other.radius_;
      return std::max(delta, 0.0);
    }

    double MinDistanceSq(
      const MetricType &metric, const BallBound& other) const {

      return core::math::Pow<2, 1>(MinDistance(metric, other));
    }

    /**
     * Computes maximum distance.
     */
    double MaxDistance(const MetricType &metric, const Point& point) const {
      return MidDistance(metric, point) + radius_;
    }

    double MaxDistanceSq(const MetricType &metric, const Point& point) const {
      return core::math::Pow<2, 1>(MaxDistance(metric, point));
    }

    /**
     * Computes maximum distance.
     */
    double MaxDistance(const MetricType &metric, const BallBound& other) const {
      return MidDistance(other.center_) + radius_ + other.radius_;
    }

    double MaxDistanceSq(
      const MetricType &metric, const BallBound& other) const {
      return core::math::Pow<2, 1>(MaxDistance(other));
    }

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    core::math::Range RangeDistance(
      const MetricType &metric, const BallBound& other) const {

      double delta = MidDistance(metric, other.center_);
      double sumradius = radius_ + other.radius_;
      return core::math::Range(
               std::max(delta - sumradius, 0.0),
               delta + sumradius);
    }

    core::math::Range RangeDistanceSq(
      const MetricType &metric,
      const BallBound& other) const {

      double delta = MidDistance(metric, other.center_);
      double sumradius = radius_ + other.radius_;
      return core::math::Range(
               core::math::Pow<2, 1>(std::max(delta - sumradius, 0.0)),
               core::math::Pow<2, 1>(delta + sumradius));
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
    double MinToMid(const MetricType &metric, const BallBound& other) const {
      double delta = MidDistance(metric, other.center_) - radius_;
      return std::max(delta, 0.0);
    }

    double MinToMidSq(const MetricType &metric, const BallBound& other) const {
      return core::math::Pow<2, 1>(MinToMid(other));
    }

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxDistance(
      const MetricType &metric, const BallBound& other) const {

      double delta =
        MidDistance(metric, other.center_) + other.radius_ - radius_;
      return std::max(delta, 0.0);
    }

    double MinimaxDistanceSq(
      const MetricType &metric, const BallBound& other) const {
      return core::math::Pow<2, 1>(MinimaxDistance(metric, other));
    }

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidDistance(const MetricType &metric, const BallBound& other) const {
      return MidDistance(metric, other.center_);
    }

    double MidDistanceSq(
      const MetricType &metric, const BallBound& other) const {
      return core::math::Pow<2, 1>(MidDistance(metric, other));
    }

    double MidDistance(
      const MetricType &metric, const Point& point) const {

      return metric.Distance(center_, point);
    }
};
};
};

#endif
