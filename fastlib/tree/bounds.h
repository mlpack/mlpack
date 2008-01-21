// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file tree/bounds.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * TODO: Come up with a better design so you can do plug-and-play distance
 * metrics.
 *
 * @experimental
 */

#ifndef TREE_BOUNDS_H
#define TREE_BOUNDS_H

#include "la/matrix.h"
#include "la/la.h"

#include "math/math.h"

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class DHrectBound {
 public:
  static const int PREFERRED_POWER = t_pow;

 private:
  DRange *bounds_;
  index_t dim_;

  OT_DEF(DHrectBound) {
    OT_MY_OBJECT(dim_);
    OT_MALLOC_ARRAY(bounds_, dim_);
  }

 public:
  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   */
  void Init(index_t dimension) {
    //DEBUG_ASSERT_MSG(dim_ == BIG_BAD_NUMBER, "Already initialized");
    bounds_ = mem::Alloc<DRange>(dimension);

    dim_ = dimension;
    Reset();
  }

  /**
   * Resets all dimensions to the empty set.
   */
  void Reset() {
    for (index_t i = 0; i < dim_; i++) {
      bounds_[i].InitEmptySet();
    }
  }

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const Vector& point) const {
    for (index_t i = 0; i < point.length(); i++) {
      if (!bounds_[i].Contains(point[i])) {
        return false;
      }
    }

    return true;
  }

  /** Gets the dimensionality */
  index_t dim() const {
    return dim_;
  }

  /**
   * Gets the range for a particular dimension.
   */
  const DRange& get(index_t i) const {
    DEBUG_BOUNDS(i, dim_);
    return bounds_[i];
  }

  /** Calculates the midpoint of the range */
  void CalculateMidpoint(Vector *centroid) const {
    centroid->Init(dim_);
    for (index_t i = 0; i < dim_; i++) {
      (*centroid)[i] = bounds_[i].mid();
    }
  }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const double *mpoint) const {
    double sum = 0;
    const DRange *mbound = bounds_;

    index_t d = dim_;

    do {
      double v = *mpoint;
      double v1 = mbound->lo - v;
      double v2 = v - mbound->hi;

      v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      mbound++;
      mpoint++;

      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    } while (--d);

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const Vector& point) const {
    DEBUG_SAME_SIZE(point.length(), dim_);
    return MinDistanceSq(point.ptr());
  }

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].lo - a[d].hi;
      double v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistanceSq(const Vector& point) const {
    double sum = 0;

    DEBUG_SAME_SIZE(point.length(), dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]);
      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(b[d].hi - a[d].lo, a[d].hi - b[d].lo);
      sum += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
  }

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  DRange RangeDistanceSq(const DHrectBound& other) const {
    double sum_lo = 0;
    double sum_hi = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].lo - a[d].hi;
      double v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      double v_lo = (v1 + fabs(v1)) + (v2 + fabs(v2));
      double v_hi = -std::min(v1, v2);

      sum_lo += math::Pow<t_pow, 1>(v_lo); // v_lo is non-negative
      sum_hi += math::Pow<t_pow, 1>(v_hi); // v_hi is non-negative
    }

    return DRange(math::Pow<2, t_pow>(sum_lo) / 4,
        math::Pow<2, t_pow>(sum_hi));
  }

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
   */
  DRange RangeDistanceSq(const Vector& point) const {
    double sum_lo = 0;
    double sum_hi = 0;
    const double *mpoint = point.ptr();
    const DRange *mbound = bounds_;

    DEBUG_SAME_SIZE(point.length(), dim_);

    index_t d = dim_;
    do {
      double v = *mpoint;
      double v1 = mbound->lo - v;
      double v2 = v - mbound->hi;

      sum_lo += math::Pow<t_pow, 1>((v1 + fabs(v1)) + (v2 + fabs(v2)));
      sum_hi += math::Pow<t_pow, 1>(-std::min(v1, v2));

      mpoint++;
      mbound++;
    } while (--d);

    return DRange(math::Pow<2, t_pow>(sum_lo) / 4,
                  math::Pow<2, t_pow>(sum_hi));
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
  double MinToMidSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = b->mid();
      double v1 = a->lo - v;
      double v2 = v - a->hi;

      v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      a++;
      b++;

      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].hi - a[d].hi;
      double v2 = a[d].lo - b[d].lo;
      double v = std::max(v1, v2);
      v = (v + fabs(v)); /* truncate negatives to zero */
      sum += math::Pow<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      sum += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Expands this region to include a new point.
   */
  DHrectBound& operator |= (const Vector& vector) {
    DEBUG_SAME_SIZE(vector.length(), dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= vector[i];
    }

    return *this;
  }

  /**
   * Expands this region to encompass another bound.
   */
  DHrectBound& operator |= (const DHrectBound& other) {
    DEBUG_SAME_SIZE(other.dim_, dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= other.bounds_[i];
    }

    return *this;
  }
};

/**
 * An L_p metric for vector spaces.
 *
 * A generic Metric class should simply compute the distance between
 * two points.  An LMetric operates for integer powers on Vector spaces.
 */
template<int t_pow>
class LMetric {
 public:
  /**
   * Computes the distance metric between two points.
   */
  static double Distance(const Vector& a, const Vector& b) {
    return math::Pow<1, t_pow>(
        la::RawLMetric<t_pow>(a.length(), a.ptr(), b.ptr()));
  }

  /**
   * Computes the distance metric between two points, raised to a
   * particular power.
   *
   * This might be faster so that you could get, for instance, squared
   * L2 distance.
   */
  template<int t_result_pow>
  static double PowDistance(const Vector& a, const Vector& b) {
    return math::Pow<t_result_pow, t_pow>(
        la::RawLMetric<t_pow>(a.length(), a.ptr(), b.ptr()));
  }
};

/**
 * Ball bound that works in arbitrary metric spaces.
 *
 * See LMetric for an example metric template parameter.
 *
 * To initialize this, set the radius with @c set_radius
 * and set the point by initializing @c point() directly.
 */
template<typename TMetric = LMetric<2>, typename TPoint = Vector>
class DBallBound {
 public:
  typedef TPoint Point;
  typedef TMetric Metric;

 private:
  double radius_;
  TPoint center_;

  OT_DEF(DBallBound) {
    OT_MY_OBJECT(radius_);
    OT_MY_OBJECT(center_);
  }

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
    ot::Copy(center_, centroid);
  }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const Point& point) const {
    return math::ClampNonNegative(MidDistance(point) - radius_);
  }

  double MinDistanceSq(const Point& point) const {
    return math::Pow<2, 1>(MinDistance(point));
  }

  /**
   * Calculates minimum bound-to-bound squared distance.
   */
  double MinDistance(const DBallBound& other) const {
    double delta = MidDistance(other.center_) - radius_ - other.radius_;
    return math::ClampNonNegative(delta);
  }

  double MinDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MinDistance(other));
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const Point& point) const {
    return MidDistance(point) + radius_;
  }

  double MaxDistanceSq(const Point& point) const {
    return math::Pow<2, 1>(MaxDistance(point));
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const DBallBound& other) const {
    return MidDistance(other.center_) + radius_ + other.radius_;
  }

  double MaxDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MaxDistance(other));
  }

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  DRange RangeDistance(const DBallBound& other) const {
    double delta = MidDistance(other.center_);
    double sumradius = radius_ + other.radius_;
    return DRange(
       math::ClampNonNegative(delta - sumradius),
       delta + sumradius);
  }

  DRange RangeDistanceSq(const DBallBound& other) const {
    double delta = MidDistance(other.center_);
    double sumradius = radius_ + other.radius_;
    return DRange(
       math::Pow<2, 1>(math::ClampNonNegative(delta - sumradius)),
       math::Pow<2, 1>(delta + sumradius));
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
  double MinToMid(const DBallBound& other) const {
    double delta = MidDistance(other.center_) - radius_;
    return math::ClampNonNegative(delta);
  }

  double MinToMidSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MinToMid(other));
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistance(const DBallBound& other) const {
    double delta = MidDistance(other.center_) + other.radius_ - radius_;
    return math::ClampNonNegative(delta);
  }

  double MinimaxDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MinimaxDistance(other));
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistance(const DBallBound& other) const {
    return MidDistance(other.center_);
  }

  double MidDistanceSq(const DBallBound& other) const {
    return math::Pow<2, 1>(MidDistance(other));
  }

  double MidDistance(const Point& point) const {
    return Metric::Distance(center_, point);
  }
};

#endif
