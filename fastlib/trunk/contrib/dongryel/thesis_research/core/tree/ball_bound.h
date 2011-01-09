/** @file tree/ball_bound.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  Bounds that are useful for binary space partitioning trees.
 */

#ifndef CORE_TREE_BALL_BOUND_H
#define CORE_TREE_BALL_BOUND_H

#include <gsl/gsl_sf_gamma.h>
#include <boost/serialization/string.hpp>
#include "core/math/math_lib.h"
#include "core/math/range.h"
#include "core/metric_kernels/abstract_metric.h"
#include "core/table/dense_point.h"

namespace core {
namespace tree {

/**
 * Ball bound that works in arbitrary metric spaces.
 */
class BallBound {

  private:
    double radius_;
    core::table::DensePoint center_;

    friend class boost::serialization::access;

  public:

    void Print() const {
      printf("Hypersphere of radius %g centered at: \n", radius_);
      for(int i = 0; i < center_.length(); i++) {
        printf("%g ", center_[i]);
      }
      printf("\n");
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & radius_;
      ar & center_;
    }

    void RandomPointInside(arma::vec *random_point_out) const {

      // First, generate $D$-dimensional Gaussian vector.
      random_point_out->set_size(center_.length());
      for(int i = 0; i < center_.length(); i++) {
        (*random_point_out)[i] = core::math::RandGaussian(1.0);
      }

      // Scale it by an appropriate factor involving the incomplete
      // Gamma function.
      double squared_length = arma::norm(*random_point_out, "fro");
      double first_number = squared_length * 0.5;
      double second_number = center_.length() * 0.5;
      double factor =
        pow(
          gsl_sf_gamma_inc_P(first_number, second_number),
          1.0 / static_cast<double>(center_.length())) / sqrt(squared_length);
      (*random_point_out) = (*random_point_out) * factor;

      // Scale the resulting vector by the radius and offset by the
      // center coordinate.
      arma::vec center_alias;
      core::table::DensePointToArmaVec(center_, &center_alias);
      (*random_point_out) = (*random_point_out) * radius_;
      (*random_point_out) += center_alias;
    }

    void RandomPointInside(core::table::DensePoint *random_point_out) const {
      random_point_out->Init(center_.length());
      arma::vec random_point_out_alias(
        random_point_out->ptr(), center_.length(), false);
      this->RandomPointInside(&random_point_out_alias);
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

    const core::table::DensePoint& center() const {
      return center_;
    }

    core::table::DensePoint& center() {
      return center_;
    }

    /**
     * Determines if a point is within this bound.
     */
    bool Contains(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::DensePoint& point) const {
      return MidDistance(metric, point) <= radius_;
    }

    /**
     * Calculates minimum bound-to-point squared distance.
     */
    double MinDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::DensePoint& point) const {

      return std::max(MidDistance(metric, point) - radius_, 0.0);
    }

    double MinDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::DensePoint& point) const {

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
      const core::table::DensePoint& point) const {
      return MidDistance(metric, point) + radius_;
    }

    double MaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::DensePoint& point) const {
      return core::math::Pow<2, 1>(MaxDistance(metric, point));
    }

    /**
     * Computes maximum distance.
     */
    double MaxDistance(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return MidDistance(metric, other.center_) + radius_ + other.radius_;
    }

    double MaxDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MaxDistance(metric, other));
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
      return core::math::Pow<2, 1>(MinToMid(metric, other));
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
      const core::table::DensePoint& point) const {
      return metric.Distance(center_, point);
    }

    double MidDistanceSq(
      const core::metric_kernels::AbstractMetric &metric,
      const core::table::DensePoint& point) const {
      return metric.DistanceSq(center_, point);
    }
};
};
};

#endif
