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

#ifndef TREE_GEN_BOUNDS_H
#define TREE_GEN_BOUNDS_H

#include "fastlib/la/matrix.h"
#include "fastlib/la/la.h"

#include "fastlib/math/math_lib.h"
#include "gen_range.h"
#include "fastlib/mmanager/memory_manager.h"

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<typename T, int t_pow = 2>
class GenHrectBound {
 public:
  static const int PREFERRED_POWER = t_pow;

 private:
  GenRange<T> *bounds_;
  index_t dim_;

  OT_DEF(GenHrectBound) {
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
    bounds_ = mem::Alloc<GenRange<T> >(dimension);

    dim_ = dimension;
    Reset();
  }

  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set statically.
   */
  void StaticInit(index_t dimension) {
    //DEBUG_ASSERT_MSG(dim_ == BIG_BAD_NUMBER, "Already initialized");
    bounds_ = 
      mmapmm::MemoryManager<false>::allocator_->Alloc<GenRange<T> >(dimension);

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
  bool Contains(const GenVector<T>& point) const {
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
  const GenRange<T>& get(index_t i) const {
    DEBUG_BOUNDS(i, dim_);
    return bounds_[i];
  }

  /** Calculates the midpoint of the range */
  void CalculateMidpoint(GenVector<T> *centroid) const {
    centroid->Init(dim_);
    for (index_t i = 0; i < dim_; i++) {
      (*centroid)[i] = bounds_[i].mid();
    }
  }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const T *mpoint) const {
    double sum = 0;
    const GenRange<T> *mbound = bounds_;

    index_t d = dim_;

    do {
      T v = *mpoint;
      T v1 = mbound->lo - v;
      T v2 = v - mbound->hi;

      v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      mbound++;
      mpoint++;

      sum += std::pow(v,t_pow); // v is non-negative
    } while (--d);

    return std::pow(sum, 2.0/t_pow) / 4;
  }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const GenVector<T>& point) const {
    DEBUG_SAME_SIZE(point.length(), dim_);
    return MinDistanceSq(point.ptr());
  }

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistanceSq(const GenHrectBound& other) const {
    double sum = 0;
    const GenRange<T> *a = this->bounds_;
    const GenRange<T> *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      T v1 = b[d].lo - a[d].hi;
      T v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      T v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      sum += std::pow(v, t_pow); // v is non-negative
    }

    return std::pow(sum, 2.0/t_pow) / 4;
  }

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistanceSq(const GenVector<T>& point) const {
    double sum = 0;

    DEBUG_SAME_SIZE(point.length(), dim_);

    for (index_t d = 0; d < dim_; d++) {
      T v = std::max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]);
      sum += std::pow(v,t_pow); // v is non-negative
    }

    return std::pow(sum, 2.0/t_pow);
  }

  /**
   * Computes maximum distance.
   */
  double MaxDistanceSq(const GenHrectBound& other) const {
    double sum = 0;
    const GenRange<T> *a = this->bounds_;
    const GenRange<T> *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      T v = std::max(b[d].hi - a[d].lo, a[d].hi - b[d].lo);
      sum += std::pow(v,t_pow); // v is non-negative
    }

    return std::pow(sum, 2.0/t_pow);
  }

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  GenRange<double> RangeDistanceSq(const GenHrectBound<T>& other) const {
    double sum_lo = 0;
    double sum_hi = 0;
    const GenRange<T> *a = this->bounds_;
    const GenRange<T> *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      T v1 = b[d].lo - a[d].hi;
      T v2 = a[d].lo - b[d].hi;
      // We invoke the following:
      //   x + fabs(x) = max(x * 2, 0)
      //   (x * 2)^2 / 4 = x^2
      T v_lo = (v1 + fabs(v1)) + (v2 + fabs(v2));
      T v_hi = -std::min(v1, v2);

      sum_lo += std::pow(v_lo,t_pow); // v_lo is non-negative
      sum_hi += std::pow(v_hi,t_pow); // v_hi is non-negative
    }

    return GenRange<double>(std::pow(sum_lo, 2.0/t_pow)/4.0,
                            std::pow(sum_hi, 2.0/t_pow));
  }

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
   */
  GenRange<double> RangeDistanceSq(const GenVector<T>& point) const {
    double sum_lo = 0;
    double sum_hi = 0;
    const T *mpoint = point.ptr();
    const GenRange<T> *mbound = bounds_;

    DEBUG_SAME_SIZE(point.length(), dim_);

    index_t d = dim_;
    do {
      T v = *mpoint;
      T v1 = mbound->lo - v;
      T v2 = v - mbound->hi;

      sum_lo += std::pow((v1 + fabs(v1)) + (v2 + fabs(v2)),t_pow);
      sum_hi += std::pow(-std::min(v1, v2),t_pow);

      mpoint++;
      mbound++;
    } while (--d);

    return GenRange<double>(std::pow(sum_lo, 2.0/t_pow)/4.0,
                            std::pow(sum_hi, 2.0/t_pow));
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
  double MinToMidSq(const GenHrectBound<T>& other) const {
    double sum = 0;
    const GenRange<T> *a = this->bounds_;
    const GenRange<T> *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      T v = b->mid();
      T v1 = a->lo - v;
      T v2 = v - a->hi;

      v = (v1 + fabs(v1)) + (v2 + fabs(v2));

      a++;
      b++;

      sum += std::pow(v,t_pow); // v is non-negative
    }

    return std::pow(sum, 2.0/t_pow) / 4.0;
  }

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistanceSq(const GenHrectBound<T>& other) const {
    double sum = 0;
    const GenRange<T> *a = this->bounds_;
    const GenRange<T> *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < mdim; d++) {
      T v1 = b[d].hi - a[d].hi;
      T v2 = a[d].lo - b[d].lo;
      T v = std::max(v1, v2);
      v = (v + fabs(v)); /* truncate negatives to zero */
      sum += std::pow(v,t_pow); // v is non-negative
    }

    return std::pow(sum, 2.0/t_pow) / 4.0;
  }

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistanceSq(const GenHrectBound<T>& other) const {
    double sum = 0;
    const GenRange<T> *a = this->bounds_;
    const GenRange<T> *b = other.bounds_;

    DEBUG_SAME_SIZE(dim_, other.dim_);

    for (index_t d = 0; d < dim_; d++) {
      sum += std::pow(a[d].hi + a[d].lo - b[d].hi - b[d].lo, t_pow);
    }

    return std::pow(sum, 2.0/t_pow) / 4.0;
  }

  /**
   * Expands this region to include a new point.
   */
  GenHrectBound& operator |= (const GenVector<T>& vector) {
    DEBUG_SAME_SIZE(vector.length(), dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= vector[i];
    }

    return *this;
  }

  /**
   * Expands this region to encompass another bound.
   */
  GenHrectBound& operator |= (const GenHrectBound<T>& other) {
    DEBUG_SAME_SIZE(other.dim_, dim_);

    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= other.bounds_[i];
    }

    return *this;
  }
};

#endif
