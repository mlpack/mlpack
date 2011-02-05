/** @file tree/ball_bound.h
 *
 *  A declaration of a ball bound.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_BALL_BOUND_H
#define CORE_TREE_BALL_BOUND_H

#include <gsl/gsl_sf_gamma.h>
#include <boost/serialization/string.hpp>
#include "core/math/math_lib.h"
#include "core/math/range.h"
#include "core/table/dense_point.h"

namespace core {
namespace tree {

/** @brief Ball bound that works in arbitrary metric spaces.
 */
class BallBound {

  private:

    /** @brief The radius of the ball bound.
     */
    double radius_;

    /** @brief The center of the ball bound.
     */
    core::table::DensePoint center_;

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief Returns whether the bound has been initialized or not.
     */
    bool is_initialized() const {
      return center_.length() > 0;
    }

    /** @brief The Assignment operator.
     */
    void operator=(const BallBound &ball_bound_in) {
      radius_ = ball_bound_in.radius();
      center_.Copy(ball_bound_in.center());
    }

    /** @brief The default constructor.
     */
    BallBound() {
      radius_ = 0;
    }

    /** @brief The copy constructor.
     */
    BallBound(const BallBound &bound_in) {
      this->operator=(bound_in);
    }

    /** @brief Resets the bound to an empty sphere centered at the
     *  origin.
     */
    void Reset() {
      radius_ = 0;
      center_.SetZero();
    }

    /** @brief Prints out the ball bound.
     */
    void Print() const {
      printf("Hypersphere of radius %g centered at: \n", radius_);
      for(int i = 0; i < center_.length(); i++) {
        printf("%g ", center_[i]);
      }
      printf("\n");
    }

    /** @brief Serialize/unserialize the radius with its center.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & radius_;
      ar & center_;
    }

    /** @brief Returns the dimensionality.
     */
    int dim() const {
      return center_.length();
    }

    /** @brief Copy the given ball bound.
     */
    void Copy(const BallBound &other_bound) {
      radius_ = other_bound.radius();
      center_.Copy(other_bound.center());
    }

    /** @brief Computes the furthest point inside the ball bound from
     *         the given point. This point is on the surface of the
     *         sphere.
     */
    template<typename PointType>
    void FurthestPoint(
      const PointType &avoid_point, arma::vec *furthest_point_out) const {

      // The vector pointing from the avoid_point to the center.
      arma::vec direction = center_ - avoid_point;
      double direction_norm = arma::norm(direction, 2);
      direction = direction / direction_norm;
      double factor = direction_norm + radius_;
      (*furthest_point_out) = avoid_point + factor * direction;
    }

    /** @brief Generate a random point within the ball bound with
     *         uniform probability.
     */
    void RandomPointInside(arma::vec *random_point_out) const {

      // First, generate $D$-dimensional Gaussian vector.
      random_point_out->set_size(center_.length());
      for(int i = 0; i < center_.length(); i++) {
        (*random_point_out)[i] = core::math::RandGaussian(1.0);
      }

      // Scale it by an appropriate factor
      double squared_length = arma::dot(*random_point_out, *random_point_out);
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

    /** @brief Generate a random point within the ball bound with
     *         uniform probability.
     */
    void RandomPointInside(core::table::DensePoint *random_point_out) const {
      random_point_out->Init(center_.length());
      arma::vec random_point_out_alias(
        random_point_out->ptr(), center_.length(), false);
      this->RandomPointInside(&random_point_out_alias);
    }

    /** @brief Initializes a ball bound with the given dimensionality.
     */
    void Init(int dimension) {
      radius_ = 0.0;
      center_.Init(dimension);
      Reset();
    }

    /** @brief Returns the radius.
     */
    double radius() const {
      return radius_;
    }

    /** @brief Sets the radius.
     */
    void set_radius(double d) {
      radius_ = d;
    }

    /** @brief Returns the centroid.
     */
    const core::table::DensePoint& center() const {
      return center_;
    }

    /** @brief Returns the centroid.
     */
    core::table::DensePoint& center() {
      return center_;
    }

    /** @brief Returns the centroid.
     */
    void center(core::table::DensePoint *center_out) const {
      center_out->Copy(center_);
    }

    /** @brief Determines if a point is within this bound.
     */
    template<typename MetricType>
    bool Contains(
      const MetricType &metric,
      const core::table::DensePoint& point) const {
      return MidDistance(metric, point) <= radius_;
    }

    /** @brief Determines whether another ball bound is entirely
     *         within this bound.
     */
    template<typename MetricType>
    bool Contains(
      const MetricType &metric,
      const core::tree::BallBound &bound_in) const {
      return MaxDistance(metric, bound_in) <= radius_;
    }

    /** @brief Calculates minimum bound-to-point squared distance.
     */
    template<typename MetricType>
    double MinDistance(
      const MetricType &metric,
      const core::table::DensePoint& point) const {

      return std::max(MidDistance(metric, point) - radius_, 0.0);
    }

    template<typename MetricType>
    double MinDistanceSq(
      const MetricType &metric,
      const core::table::DensePoint& point) const {

      return core::math::Pow<2, 1>(MinDistance(metric, point));
    }

    /** @brief Calculates minimum bound-to-bound squared distance.
     */
    template<typename MetricType>
    double MinDistance(
      const MetricType &metric,
      const BallBound& other) const {
      double delta =
        MidDistance(metric, other.center_) - radius_ - other.radius_;
      return std::max(delta, 0.0);
    }

    template<typename MetricType>
    double MinDistanceSq(
      const MetricType &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MinDistance(metric, other));
    }

    /** @brief Computes maximum distance.
     */
    template<typename MetricType>
    double MaxDistance(
      const MetricType &metric,
      const core::table::DensePoint& point) const {
      return MidDistance(metric, point) + radius_;
    }

    template<typename MetricType>
    double MaxDistanceSq(
      const MetricType &metric,
      const core::table::DensePoint& point) const {
      return core::math::Pow<2, 1>(MaxDistance(metric, point));
    }

    /** @brief Computes maximum distance.
     */
    template<typename MetricType>
    double MaxDistance(
      const MetricType &metric,
      const BallBound& other) const {
      return MidDistance(metric, other.center_) + radius_ + other.radius_;
    }

    template<typename MetricType>
    double MaxDistanceSq(
      const MetricType &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MaxDistance(metric, other));
    }

    /** @brief Calculates minimum and maximum bound-to-bound squared
     *  distance.
     *
     * Example: bound1.MinDistanceSq(other) for minimum squared distance.
     */
    template<typename MetricType>
    core::math::Range RangeDistance(
      const MetricType &metric,
      const BallBound& other) const {

      double delta = MidDistance(metric, other.center_);
      double sumradius = radius_ + other.radius_;
      return core::math::Range(
               std::max(delta - sumradius, 0.0),
               delta + sumradius);
    }

    template<typename MetricType>
    core::math::Range RangeDistanceSq(
      const MetricType &metric,
      const BallBound& other) const {

      double delta = MidDistance(metric, other.center_);
      double sumradius = radius_ + other.radius_;
      return core::math::Range(
               core::math::Pow<2, 1>(std::max(delta - sumradius, 0.0)),
               core::math::Pow<2, 1>(delta + sumradius));
    }

    /** @brief Calculates closest-to-their-midpoint bounding box
     * distance, i.e. calculates their midpoint and finds the minimum
     * box-to-point distance.
     *
     * Equivalent to:
     * <code>
     * other.CalcMidpoint(&other_midpoint)
     * return MinDistanceSqToPoint(other_midpoint)
     * </code>
     */
    template<typename MetricType>
    double MinToMid(
      const MetricType &metric,
      const BallBound& other) const {
      double delta = MidDistance(metric, other.center_) - radius_;
      return std::max(delta, 0.0);
    }

    template<typename MetricType>
    double MinToMidSq(
      const MetricType &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MinToMid(metric, other));
    }

    /** @brief Computes minimax distance, where the other node is
     *         trying to avoid me.
     */
    template<typename MetricType>
    double MinimaxDistance(
      const MetricType &metric,
      const BallBound& other) const {
      double delta =
        MidDistance(metric, other.center_) + other.radius_ - radius_;
      return std::max(delta, 0.0);
    }

    template<typename MetricType>
    double MinimaxDistanceSq(
      const MetricType &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MinimaxDistance(metric, other));
    }

    /**
     * Calculates midpoint-to-midpoint bounding box distance.
     */
    template<typename MetricType>
    double MidDistance(
      const MetricType &metric,
      const BallBound& other) const {
      return MidDistance(metric, other.center_);
    }

    template<typename MetricType>
    double MidDistanceSq(
      const MetricType &metric,
      const BallBound& other) const {
      return core::math::Pow<2, 1>(MidDistance(metric, other));
    }

    template<typename MetricType>
    double MidDistance(
      const MetricType &metric,
      const core::table::DensePoint& point) const {
      return metric.Distance(center_, point);
    }

    template<typename MetricType>
    double MidDistanceSq(
      const MetricType &metric,
      const core::table::DensePoint& point) const {
      return metric.DistanceSq(center_, point);
    }

    /** @brief Expands this region to include a new point.
     */
    template<typename MetricType>
    BallBound& Expand(
      const MetricType &metric, const core::table::DensePoint &vector) {

      // If the point is already inside the sphere, then no expansion
      // is necessary.
      if(! this->Contains(metric, vector)) {

        // Compute the furthest point from the new point.
        arma::vec furthest_point;
        this->FurthestPoint(vector, &furthest_point);

        // The center is the mid way between the new point and the
        // furthest point.
        arma::vec center_alias;
        core::table::DensePointToArmaVec(center_, &center_alias);
        arma::vec vector_alias;
        core::table::DensePointToArmaVec(vector, &vector_alias);
        center_alias = 0.5 * (vector_alias + furthest_point);
        radius_ = metric.Distance(center_, vector);
      }
      return *this;
    }

    /** @brief Expands this region to encompass another bound.
     */
    template<typename MetricType>
    BallBound& Expand(
      const MetricType &metric, const BallBound &other) {

      // If the given ball bound is already contained, then no need to
      // expand.
      if(! this->Contains(metric, other)) {

        // Compute the furthest point from each center.
        arma::vec furthest_point_on_self;
        arma::vec furthest_point_on_new_bound;
        this->FurthestPoint(other.center(), &furthest_point_on_self);
        other.FurthestPoint(center_, &furthest_point_on_new_bound);

        // The center is the midway between two furthest points.
        arma::vec center_alias;
        core::table::DensePointToArmaVec(center_, &center_alias);
        center_alias = 0.5 * (
                         furthest_point_on_self + furthest_point_on_new_bound);
        radius_ = metric.Distance(center_, furthest_point_on_self);
      }
      return *this;
    }
};
}
}

#endif
