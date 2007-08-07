// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @param bounds.h
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

/**
 * Simple real-valued range.
 *
 * @experimental
 */
struct DRange {
 public:
  /**
   * The lower bound.
   */
  double lo;
  /**
   * The upper bound.
   */
  double hi;
  
  OT_DEF_BASIC(DRange) {
    OT_MY_OBJECT(lo);
    OT_MY_OBJECT(hi);
  }
  
 public:
  /** Initializes to specified values. */
  DRange(double lo_in, double hi_in)
      : lo(lo_in), hi(hi_in)
      {}

  /** Initialize to an empty set, where lo > hi. */
  void InitEmptySet() {
    lo = DBL_MAX;
    hi = -DBL_MAX;
  }

  /** Initializes to -infinity to infinity. */
  void InitUniversalSet() {
    lo = -DBL_MAX;
    hi = DBL_MAX;
  }
  
  /** Initializes to a range of values. */
  void Init(double lo_in, double hi_in) {
    lo = lo_in;
    hi = hi_in;
  }

  /**
   * Resets to a range of values.
   *
   * Since there is no dynamic memory this is the same as Init, but calling
   * Reset instead of Init probably looks more similar to surrounding code.
   */
  void Reset(double lo_in, double hi_in) {
    lo = lo_in;
    hi = hi_in;
  }

  /**
   * Gets the span of the range, hi - lo.
   */  
  double width() const {
    return hi - lo;
  }

  /**
   * Gets the midpoint of this range.
   */  
  double mid() const {
    return (hi + lo) / 2;
  }
  
  /**
   * Interpolates (factor) * hi + (1 - factor) * lo.
   */
  double interpolate(double factor) const {
    return factor * width() + lo;
  }

  /**
   * Simulate an union by growing the range if necessary.
   */
  const DRange& operator |= (double d) {
    if (unlikely(d < lo)) {
      lo = d;
    }
    if (unlikely(d > hi)) {
      hi = d;
    }
    return *this;
  }

  /**
   * Sets this range to include only the specified value, or
   * becomes an empty set if the range does not contain the number.
   */
  const DRange& operator &= (double d) {
    if (likely(d > lo)) {
      lo = d;
    }
    if (likely(d < hi)) {
      hi = d;
    }
    return *this;
  }

  /**
   * Expands range to include the other range.
   */
  const DRange& operator |= (const DRange& other) {
    if (unlikely(other.lo < lo)) {
      lo = other.lo;
    }
    if (unlikely(other.hi > hi)) {
      hi = other.hi;
    }
    return *this;
  }
  
  /**
   * Shrinks range to be the overlap with another range, becoming an empty
   * set if there is no overlap.
   */
  const DRange& operator &= (const DRange& other) {
    if (unlikely(other.lo > lo)) {
      lo = other.lo;
    }
    if (unlikely(other.hi < hi)) {
      hi = other.hi;
    }
    return *this;
  }
  
  /** Sums the upper and lower independently. */
  const DRange& operator += (const DRange& other) {
    lo += other.lo;
    hi += other.hi;
    return *this;
  }
  
  /** Subtracts from the upper and lower independently. */
  const DRange& operator -= (const DRange& other) {
    lo -= other.lo;
    hi -= other.hi;
    return *this;
  }
  
  /** Adds to the upper and lower independently. */
  const DRange& operator += (double d) {
    lo += d;
    hi += d;
    return *this;
  }
  
  /** Subtracts from the upper and lower independently. */
  const DRange& operator -= (double d) {
    lo -= d;
    hi -= d;
    return *this;
  }

  friend DRange operator + (const DRange& a, const DRange& b) {
    DRange result;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi;
    return result;
  }

  friend DRange operator - (const DRange& a, const DRange& b) {
    DRange result;
    result.lo = a.lo - b.lo;
    result.hi = a.hi - b.hi;
    return result;
  }
  
  friend DRange operator + (const DRange& a, double b) {
    DRange result;
    result.lo = a.lo + b;
    result.hi = a.hi + b;
    return result;
  }

  friend DRange operator - (const DRange& a, double b) {
    DRange result;
    result.lo = a.lo - b;
    result.hi = a.hi - b;
    return result;
  }

  /**
   * Takes the maximum of upper and lower bounds independently.
   */
  void MaxWith(const DRange& range) {
    if (unlikely(range.lo > lo)) {
      lo = range.lo;
    }
    if (unlikely(range.hi > hi)) {
      hi = range.hi;
    }
  }
  
  /**
   * Takes the minimum of upper and lower bounds independently.
   */
  void MinWith(const DRange& range) {
    if (unlikely(range.lo < lo)) {
      lo = range.lo;
    }
    if (unlikely(range.hi < hi)) {
      hi = range.hi;
    }
  }

  /**
   * Takes the maximum of upper and lower bounds independently.
   */
  void MaxWith(double v) {
    if (unlikely(v > lo)) {
      lo = v;
      if (unlikely(v > hi)) {
        hi = v;
      }
    }
  }
  
  /**
   * Takes the minimum of upper and lower bounds independently.
   */
  void MinWith(double v) {
    if (unlikely(v < hi)) {
      hi = v;
      if (unlikely(v < lo)) {
        lo = v;
      }
    }
  }

  /**
   * Compares if this is STRICTLY less than another range.
   */  
  friend bool operator < (const DRange& a, const DRange& b) {
    return a.hi < b.lo;
  }
  /**
   * Compares if this is STRICTLY equal to another range.
   */  
  friend bool operator == (const DRange& a, const DRange& b) {
    return a.lo == b.lo && a.hi == b.hi;
  }
  DEFINE_ALL_COMPARATORS(DRange);
  
  /**
   * Compares if this is STRICTLY less than a value.
   */  
  friend bool operator < (const DRange& a, double b) {
    return a.hi < b;
  }
  /**
   * Compares if a value is STRICTLY less than this range.
   */  
  friend bool operator < (double a, const DRange& b) {
    return a < b.lo;
  }
  DEFINE_INEQUALITY_COMPARATORS_HETERO(DRange, double);

  /**
   * Determines if a point is contained within the range.
   */  
  bool Contains(double d) const {
    return d >= lo || d <= hi;
  }
};

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 *
 * @experimental
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
   * Calculates minimum bound-to-point squared distance,
   * to the specified power.
   */
  double MinDistanceSq(const Vector& point) const {
    DEBUG_ASSERT(point.length() == dim_);
    double sumsq = 0;
    const double *mpoint = point.ptr();
    const DRange *mbound = bounds_;
    
    index_t d = dim_;
    
    do {
      double v = *mpoint;
      double v1 = mbound->lo - v;
      double v2 = v - mbound->hi;
      
      v = (v1 + fabs(v1)) + (v2 + fabs(v2));
      
      mbound++;
      mpoint++;
      
      sumsq += math::Pow<t_pow, 1>(v);
    } while (--d);
    
    return math::Pow<2, t_pow>(sumsq) / 4;
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
    double sumsq = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = b->mid();
      double v1 = a->lo - v;
      double v2 = v - a->hi;
      
      v = (v1 + fabs(v1)) + (v2 + fabs(v2));
      
      a++;
      b++;
      
      sumsq += math::Pow<t_pow, 1>(v);
    }

    return math::Pow<2, t_pow>(sumsq) / 4;
  }

  /**
   * Calculates maximum bound-to-point squared distance,
   * to the specified power.
   */
  double MaxDistanceSq(const Vector& point) const {
    double sumsq = 0;

    DEBUG_ASSERT(point.length() == dim_);

    for (index_t d = 0; d < dim_; d++) {
      sumsq += math::Pow<t_pow, 1>(
          max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]));
    }

    return math::Pow<2, t_pow>(sumsq);
  }

  /**
   * Calculates minimum bound-to-point squared distance,
   * to the specified power.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistanceSq(const DHrectBound& other) const {
    double sumsq = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_INT(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].lo - a[d].hi;
      double v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      sumsq += math::Pow<t_pow, 1>(v);
    }

    return math::Pow<2, t_pow>(sumsq) / 4;
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me,
   * to the specified power.
   */
  double MinimaxDistanceSq(const DHrectBound& other) const {
    double sumsq = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;
    
    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].hi - a[d].hi;
      double v2 = a[d].lo - b[d].lo;
      double v = max(v1, v2);
      v = (v + fabs(v)); /* truncate negatives to zero */
      sumsq += math::Pow<t_pow, 1>(v);
    }

    return math::Pow<2, t_pow>(sumsq) / 4;
  }

  /**
   * Computes maximum distance,
   * to the specified power.
   */
  double MaxDistanceSq(const DHrectBound& other) const {
    double sumsq = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < dim_; d++) {
      sumsq += math::Pow<t_pow, 1>(
          max(b[d].hi - a[d].lo, a[d].hi - b[d].lo));
    }

    return math::Pow<2, t_pow>(sumsq);
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance,
   * to the specified power.
   */
  double MidDistanceSq(const DHrectBound& other) const {
    double sumsq = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < dim_; d++) {
      sumsq += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
    }

    return math::Pow<2, t_pow>(sumsq) / 4;
  }
  
  /**
   * Expands this region to include a new point.
   */
  DHrectBound& operator |= (const Vector& vector) {
    DEBUG_SAME_INT(vector.length(), dim_);
    
    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= vector[i];
    }
    
    return *this;
  }

  /**
   * Expands this region to encompass another bound.
   */
  DHrectBound& operator |= (const DHrectBound& other) {
    DEBUG_SAME_INT(other.dim_, dim_);
    
    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= other.bounds_[i];
    }
    
    return *this;
  }
};

// Here's an idea of what ball-trees might look like.
// /**
//  * Euclidean metric for use with ball bounds.
//  *
//  * @experimental
//  */
// class DEuclideanMetric {
//  public:
//   static double CalculateMetric(const Vector& a, const Vector& b) {
//     return sqrt(la::DistanceSqEuclidean(a.length(), a.ptr(), b.ptr()));
//   }
// };
// 
// /**
//  * Bound of a ball tree.
//  *
//  * @experimental
//  */
// template<class TPoint, class TMetric>
// class BallBound {
//   FORBID_COPY(BallBound);
//   
//  public:
//   typedef TMetric Metric;
//   typedef TPoint Point;
//   
//  private:
//   Point center_;
//   double radius_;
//   
//  public:
//   BallBound() {}
//   
//   const Point& center() const {
//     return center;
//   }
//   
//   Point& center() {
//     return center;
//   }
//   
//   double radius() const {
//     return radius;
//   }
//   
//   void set_radius(double d) {
//     radius = d;
//   }
//   
//   double DistanceToCenter(const Point& point) {
//     return Metric::CalculateMetric(point, center_);
//   }
//   
//   bool Belongs(const Point& point) {
//     return DistanceToCenter(point) <= radius_;
//   }
//   
//   double MinDistanceToPoint(const Point& point) {
//     return max(0.0, DistanceToCenter(point) - radius_);
//   }
//   
//   double MaxDistanceToPoint(const Point& point) {
//     return DistanceToCenter(point) + radius_;
//   }
//   
//   double MinDistanceToBound(const BallBound& ball) {
//     return max(0,
//         DistanceToCenter(ball.center_) - (radius_ + ball.radius_));
//   }
//   
//   double MaxDistanceToBound(const BallBound& ball) {
//     return DistanceToCenter(ball.center_) + (radius_ + ball.radius_);
//   }
//   
//   double MidDistanceToBound(const BallBound& other) {
//     return DistanceToCenter(other.center_);
//   }
//   
//   double MidDistanceToPoint(const Point& point) {
//     return DistanceToCenter(point);
//   }
// };
// 
// typedef BallBound<Vector, DEuclideanMetric> DEuclideanBallBound;


#endif
