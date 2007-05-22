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

#ifndef TREE_SPBOUNDS_H
#define TREE_SPBOUNDS_H

#include "la/matrix.h"
#include "la/la.h"

/**
 * A 
 */
template<typename TValue>
class MinMaxVal {
 public:
  typedef TValue Value;
  
 public:
  Value val;
  
  OT_DEF(MinMaxVal) {
    OT_MY_OBJECT(val);
  }
  
 public:
  MinMaxVal(Value val_in) : val(val_in) {}
  MinMaxVal() {}
  
  operator Value() const { return val; }
  
  const Value& operator = (Value val_in) {
    return (val = val_in);
  }
  
  void MinWith(Value incoming_val) {
    if (unlikely(incoming_val < val)) {
      val = incoming_val;
    }
  }
  
  void MaxWith(Value incoming_val) {
    if (unlikely(incoming_val > val)) {
      val = incoming_val;
    }
  }
};

class HiBound {
};

/**
 * Simple real-valued range.
 *
 * @experimental
 */
struct SpRange {
 public:
  double lo;
  double hi;
  
  OT_DEF(SpRange) {
    OT_MY_OBJECT(lo);
    OT_MY_OBJECT(hi);
  }
  
 public:
  SpRange() {}
  SpRange(double lo_in, double hi_in)
      : lo(lo_in), hi(hi_in)
      {}
  
  void InitEmptySet() {
    lo = DBL_MAX;
    hi = -DBL_MAX;
  }
  
  void InitUniversalSet() {
    lo = -DBL_MAX;
    hi = DBL_MAX;
  }
  
  void Init(double lo_in, double hi_in) {
    lo = lo_in;
    hi = hi_in;
  }
  
  double width() const {
    return hi - lo;
  }
  
  double mid() const {
    return (hi + lo) / 2;
  }
  
  const SpRange& operator |= (double d) {
    if (unlikely(d < lo)) {
      lo = d;
    }
    if (unlikely(d > hi)) {
      hi = d;
    }
    return *this;
  }
  
  const SpRange& operator &= (double d) {
    if (unlikely(d > lo)) {
      lo = d;
    }
    if (unlikely(d < hi)) {
      hi = d;
    }
    return *this;
  }
  
  const SpRange& operator |= (const SpRange& other) {
    if (unlikely(other.lo < lo)) {
      lo = other.lo;
    }
    if (unlikely(other.hi > hi)) {
      hi = other.hi;
    }
    return *this;
  }
  
  const SpRange& operator &= (const SpRange& other) {
    if (unlikely(other.lo > lo)) {
      lo = other.lo;
    }
    if (unlikely(other.hi < hi)) {
      hi = other.hi;
    }
    return *this;
  }
  
  /** Accumulates a bound difference. */
  const SpRange& operator += (const SpRange& other) {
    lo += other.lo;
    hi += other.hi;
    return *this;
  }
  
  /** Reverses a bound difference. */
  const SpRange& operator -= (const SpRange& other) {
    lo -= other.lo;
    hi -= other.hi;
    return *this;
  }
  
  /** Uniformly increases both lower and upper bounds. */
  const SpRange& operator += (double d) {
    lo += d;
    hi += d;
    return *this;
  }
  
  /** Uniformly decreases both upper and lower bounds. */
  const SpRange& operator -= (double d) {
    lo -= d;
    hi -= d;
    return *this;
  }
  
  friend SpRange operator + (const SpRange& a, const SpRange& b) {
    SpRange result;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi;
    return result;
  }

  friend SpRange operator - (const SpRange& a, const SpRange& b) {
    SpRange result;
    result.lo = a.lo - b.lo;
    result.hi = a.hi - b.hi;
    return result;
  }
  
  friend SpRange operator + (const SpRange& a, double b) {
    SpRange result;
    result.lo = a.lo + b;
    result.hi = a.hi + b;
    return result;
  }

  friend SpRange operator - (const SpRange& a, double b) {
    SpRange result;
    result.lo = a.lo - b;
    result.hi = a.hi - b;
    return result;
  }
  
  friend bool operator < (const SpRange& a, const SpRange& b) {
    return a.hi < b.lo;
  }
  friend bool operator == (const SpRange& a, const SpRange& b) {
    return a.lo == b.lo && a.hi == b.hi;
  }
  DEFINE_ALL_COMPARATORS(SpRange);
  
  friend bool operator < (const SpRange& a, double b) {
    return a.hi < b;
  }
  friend bool operator < (double a, const SpRange& b) {
    return a < b.lo;
  }
  DEFINE_INEQUALITY_COMPARATORS_HETERO(SpRange, double);
  
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
class SpHrectBound {
 private:
  SpRange *bounds_;
  //double diagonal_sq_;
  index_t dim_;

  OT_DEF(SpHrectBound) {
    OT_MY_OBJECT(dim_);
    OT_MALLOC_ARRAY(bounds_, dim_);
  }
  
 public:
  SpHrectBound() {
    DEBUG_POISON_PTR(bounds_);
    DEBUG_ONLY(dim_ = BIG_BAD_NUMBER);
  }
  ~SpHrectBound() {
    mem::Free(bounds_);
  }

  void Init(index_t dimension) {
    DEBUG_ASSERT_MSG(dim_ == BIG_BAD_NUMBER, "Already initialized");

    bounds_ = mem::Alloc<SpRange>(dimension);

    for (index_t i = 0; i < dimension; i++) {
      bounds_[i].InitEmptySet();
    }

    dim_ = dimension;
  }
  
  bool Contains(const Vector& point) const {
    for (index_t i = 0; i < point.length(); i++) {
      if (!bounds_[i].Contains(point[i])) {
        return false;
      }
    }
    
    return true;
  }
  
  double MinDistanceSqToPoint(const Vector& point) const {
    DEBUG_ASSERT(point.length() == dim_);
    double sumsq = 0;
    const double *mpoint = point.ptr();
    const SpRange *mbound = bounds_;
    
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
  
  double MaxDistanceSqToPoint(const Vector& point) const {
    double sumsq = 0;

    DEBUG_ASSERT(point.length() == dim_);

    for (index_t d = 0; d < dim_; d++) {
      sumsq += math::Pow<t_pow, 1>(
          max(point[d] - bounds_[d].lo, bounds_[d].hi - point[d]));
    }

    return math::Pow<2, t_pow>(sumsq);
  }
  
  double MinDistanceSqToBound(const SpHrectBound& other) const {
    double sumsq = 0;
    const SpRange *a = this->bounds_;
    const SpRange *b = other.bounds_;
    index_t mdim = dim_;

    DEBUG_ASSERT(dim_ == other.dim_);


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

  double MinDistanceSqToBoundFarEnd(const SpHrectBound& other) const {
    double sumsq = 0;
    const SpRange *a = this->bounds_;
    const SpRange *b = other.bounds_;
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

  double MaxDistanceSqToBound(const SpHrectBound& other) const {
    double sumsq = 0;
    const SpRange *a = this->bounds_;
    const SpRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < dim_; d++) {
      sumsq += math::Pow<t_pow, 1>(
          max(b[d].hi - a[d].lo, a[d].hi - b[d].lo));
    }

    return math::Pow<2, t_pow>(sumsq);
  }

  double MidDistanceSqToBound(const SpHrectBound& other) const {
    double sumsq = 0;
    const SpRange *a = this->bounds_;
    const SpRange *b = other.bounds_;

    DEBUG_ASSERT(dim_ == other.dim_);
    
    for (index_t d = 0; d < dim_; d++) {
      sumsq += math::PowAbs<t_pow, 1>(a[d].hi + a[d].lo - b[d].hi - b[d].lo);
    }

    return math::Pow<2, t_pow>(sumsq) / 4;
  }
  
  SpHrectBound& operator |= (const Vector& vector) {
    DEBUG_SAME_INT(vector.length(), dim_);
    
    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= vector[i];
    }
    
    return *this;
  }

  SpHrectBound& operator |= (const SpHrectBound& other) {
    DEBUG_SAME_INT(other.dim_, dim_);
    
    for (index_t i = 0; i < dim_; i++) {
      bounds_[i] |= other.bounds_[i];
    }
    
    return *this;
  }
  
  const SpRange& get(index_t i) const {
    DEBUG_BOUNDS(i, dim_);
    return bounds_[i];
  }
};

#endif
