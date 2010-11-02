/** @file tree/ball_bound.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  Bounds that are useful for binary space partitioning trees.
 */

#ifndef CORE_TREE_BALL_BOUND_H
#define CORE_TREE_BALL_BOUND_H

#include "boost/serialization/string.hpp"
#include "core/math/math_lib.h"
#include "core/math/range.h"
#include "core/metric_kernels/abstract_metric.h"
#include "core/table/dense_point.h"

namespace core {
namespace tree {

/**
 * Ball bound that works in arbitrary metric spaces.
 */
template<typename TPoint>
class BallBound {
  public:
    typedef TPoint Point;

  private:
    double radius_;
    TPoint center_;

    friend class boost::serialization::access;

  public:

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & radius_;
      ar & center_;
    }

    void Init(int dimension) {
      radius_ = 0.0;
      center_.Init(dimension);
    }

    double radius() const {
      return radius_;
    }

    double &radius() {
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
    bool Contains(const core::table::AbstractPoint& point) const {
      return MidDistance(point) <= radius_;
    }

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {

      return std::max(MidDistance(metric, point) - radius_, 0.0);
    }

    double MinDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {

      return core::math::Pow<2, 1>(MinDistance(metric, point));
    }

    /**
     * Calculates minimum bound-to-bound squared distance.
     */
    double MinDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      double delta =
        MidDistance(metric, other.center_) - radius_ - other.radius_;
      return std::max(delta, 0.0);
    }

    double MinDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MinDistance(metric, other));
    }

    /**
     * Computes maximum distance.
     */
    double MaxDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {
      return MidDistance(metric, point) + radius_;
    }

    double MaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {
      return core::math::Pow<2, 1>(MaxDistance(metric, point));
    }

    /**
     * Computes maximum distance.
     */
    double MaxDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return MidDistance(other.center_) + radius_ + other.radius_;
    }

    double MaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MaxDistance(other));
    }

    /**
     * Calculates minimum and maximum bound-to-bound squared distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    core::math::Range RangeDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {

      double delta = MidDistance(metric, other.center_);
      double sumradius = radius_ + other.radius_;
      return core::math::Range(
               std::max(delta - sumradius, 0.0),
               delta + sumradius);
    }

    core::math::Range RangeDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
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
    double MinToMid(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      double delta = MidDistance(metric, other.center_) - radius_;
      return std::max(delta, 0.0);
    }

    double MinToMidSq(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MinToMid(other));
    }

    /**
     * Computes minimax distance, where the other node is trying to avoid me.
     */
    double MinimaxDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      double delta =
        MidDistance(metric, other.center_) + other.radius_ - radius_;
      return std::max(delta, 0.0);
    }

    double MinimaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MinimaxDistance(metric, other));
    }

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    double MidDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return MidDistance(metric, other.center_);
    }

    double MidDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MidDistance(metric, other));
    }

    double MidDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::AbstractPoint& point) const {
      return metric.Distance(center_, point);
    }
};
};
};

#endif
