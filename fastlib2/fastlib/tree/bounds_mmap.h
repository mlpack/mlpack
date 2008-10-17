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

#ifndef TREE_BOUNDS_MMAP_H
#define TREE_BOUNDS_MMAP_H

#include "fastlib/la/matrix.h"
#include "fastlib/la/la.h"
#include "fastlib/mmanager/memory_manager.h"

#include "fastlib/math/math_lib.h"
#include "bounds.h"

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class DHrectBoundMmap {
 public:
  static const int PREFERRED_POWER = t_pow;
  typedef DHrectBound<t_pow> StaticBound;
 private:
  DRange *bounds_;
  index_t dim_;

  OT_DEF(DHrectBoundMmap) {
    OT_MY_OBJECT(dim_);
    OT_MALLOC_ARRAY(bounds_, dim_);
  }

 public:
  static void* operator new(size_t size) {
    return  mmapmm::MemoryManager<false>::allocator_->Alloc<DHrectBoundMmap>();
  }
  static void operator delete(void *) {
  
  }
  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   */
  void Init(index_t dimension) {
    //DEBUG_ASSERT_MSG(dim_ == BIG_BAD_NUMBER, "Already initialized");
    bounds_ = mmapmm::MemoryManager<false>::allocator_->Alloc<DRange>(dimension);

    dim_ = dimension;
    Reset();
  }
  void Copy(DHrectBound<t_pow> & bound) {
    for(index_t i=0; i<dim_; i++) {
      bounds_[i]=bound.get(i);
    }
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
  double MinDistanceSq(const DHrectBoundMmap& other) const {
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
  * Calcualtes minimum bound-to-bound squared distance, with
  * an offset between their respective coordinate systems.
  */
 double MinDistanceSq(const DHrectBoundMmap& other, const Vector& offset) const {
   double sum = 0;
   const DRange *a = this->bounds_;
   const DRange *b = other.bounds_;
   index_t mdim = dim_;
   
   DEBUG_SAME_SIZE(dim_, other.dim_);
   //Add Debug for offset vector

   for (index_t d = 0; d < mdim; d++) {
     double v1 = b[d].lo - offset[d] - a[d].hi;
     double v2 = a[d].lo + offset[d] - b[d].lo;

     double v = (v1 + fabs(v1)) + (v2 + fabs(v2));

     sum += math::Pow<t_pow, 1>(v);
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
  double MaxDistanceSq(const DHrectBoundMmap& other) const {
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
  * Computes maximum distance with offset
  */
 double MaxDistanceSq(const DHrectBoundMmap& other, const Vector& offset) const {
    double sum = 0;
    const DRange *a = this->bounds_;
    const DRange *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      double v = std::max(b[d].hi - offset[d] - a[d].lo, 
			  a[d].hi + offset[d] - b[d].lo);
      sum += math::PowAbs<t_pow, 1>(v); // v is non-negative
    }

    return math::Pow<2, t_pow>(sum);
 }

 double PeriodicMinDistanceSq(const DHrectBoundMmap& other, const Vector& box_size)
 const {
   double sum = 0;
   const DRange *a = this->bounds_;
   const DRange *b = other.bounds_;
   
   DEBUG_SAME_SIZE(dim_, other.dim_);
   
   for (index_t d = 0; d < dim_; d++){
     double v = 0;
     bool i,j,k,l;
     i = a[d].lo < a[d].hi;
     j = b[d].lo < b[d].hi;
     k = a[d].hi > b[d].lo;
     l = b[d].hi > a[d].lo;
     v = ((i^j) & !(k | l)) * std::min(a[d].lo - b[d].hi, b[d].lo - a[d].hi);
     v = v + (i & j & (k ^ l)) * std::min(a[d].lo - b[d].hi + l*box_size[d], 
					    b[d].lo - a[d].hi + k*box_size[d]);
     sum += math::PowAbs<t_pow, 1>(v);
   }
   
   return math::Pow<2, t_pow>(sum);
 }


  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  DRange RangeDistanceSq(const DHrectBoundMmap& other) const {
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
  double MinToMidSq(const DHrectBoundMmap& other) const {
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
  double MinimaxDistanceSq(const DHrectBoundMmap& other) const {
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
  double MidDistanceSq(const DHrectBoundMmap& other) const {
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
  DHrectBoundMmap& operator |= (const Vector& vector) {
    DEBUG_SAME_SIZE(vector.length(), dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= vector[i];
    }

    return *this;
  }

  /**
   * Expands this region to encompass another bound.
   */
  DHrectBoundMmap& operator |= (const DHrectBoundMmap& other) {
    DEBUG_SAME_SIZE(other.dim_, dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= other.bounds_[i];
    }

    return *this;
  }
};



#endif
