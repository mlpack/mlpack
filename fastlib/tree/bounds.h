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
   * Calculates minimum bound-to-point squared distance,
   * to the specified power.
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
      
      sum += math::Pow<t_pow, 1>(v);
    } while (--d);
    
    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates minimum bound-to-point squared distance,
   * to the specified power.
   */
  double MinDistanceSq(const Vector& point) const {
    DEBUG_ASSERT(point.length() == dim_);
    return MinDistanceSq(point.ptr());
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

    DEBUG_ASSERT(dim_ == other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = b->mid();
      double v1 = a->lo - v;
      double v2 = v - a->hi;
      
      v = (v1 + fabs(v1)) + (v2 + fabs(v2));
      
      a++;
      b++;
      
      sum += math::Pow<t_pow, 1>(v);
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates maximum bound-to-point squared distance,
   * to the specified power.
   */
  double MaxDistanceSq(const Vector& point) const {
    double sum = 0;

    DEBUG_ASSERT(point.length() == dim_);

    for (index_t d = 0; d < dim_; d++) {
      sum += math::Pow<t_pow, 1>(
          max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]));
    }

    return math::Pow<2, t_pow>(sum);
  }

  /**
   * Calculates minimum bound-to-bound squared distance,
   * to the specified power.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistanceSq(const DHrectBound& other) const {
    double sum = 0;
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

      sum += math::Pow<t_pow, 1>(v);
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Calculates minimum and maximum bound-to-bound squared distance,
   * to the specified power.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  DRange RangeDistanceSq(const DHrectBound& other) const {
    double sum_lo = 0;
    double sum_hi = 0;
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
      double v_lo = (v1 + fabs(v1)) + (v2 + fabs(v2));
      double v_hi = min(v1, v2);

      sum_lo += math::Pow<t_pow, 1>(v_lo);
      sum_hi += math::Pow<t_pow, 1>(v_hi);
    }

    return DRange(math::Pow<2, t_pow>(sum_lo) / 4,
        math::Pow<2, t_pow>(sum_hi));
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me,
   * to the specified power.
   */
  double MinimaxDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;
    index_t mdim = dim_;
    
    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < mdim; d++) {
      double v1 = b[d].hi - a[d].hi;
      double v2 = a[d].lo - b[d].lo;
      double v = max(v1, v2);
      v = (v + fabs(v)); /* truncate negatives to zero */
      sum += math::Pow<t_pow, 1>(v);
    }

    return math::Pow<2, t_pow>(sum) / 4;
  }

  /**
   * Computes maximum distance,
   * to the specified power.
   */
  double MaxDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < dim_; d++) {
      sum += math::Pow<t_pow, 1>(
          max(b[d].hi - a[d].lo, a[d].hi - b[d].lo));
    }

    return math::Pow<2, t_pow>(sum);
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance,
   * to the specified power.
   */
  double MidDistanceSq(const DHrectBound& other) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < dim_; d++) {
      sum += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
    }

    return math::Pow<2, t_pow>(sum) / 4;
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
