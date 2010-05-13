// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file math.h
 *
 * Includes all basic FASTlib non-vector math utilities.
 */

#ifndef GEN_RANGE_H
#define GEN_RANGE_H

#include "fastlib/base/base.h"

#include <math.h>

/**
 * Simple real-valued range.
 *
 * @experimental
 */
template<typename T>
class GenRange {
 public:
  /**
   * The lower bound.
   */
  T lo;
  /**
   * The upper bound.
   */
  T hi;
  
  OT_DEF_BASIC(GenRange) {
    OT_MY_OBJECT(lo);
    OT_MY_OBJECT(hi);
  }
  
 public:
  /** Initializes to specified values. */
  GenRange(T lo_in, T hi_in)
      : lo(lo_in), hi(hi_in)
      {}

  /** Initialize to an empty set, where lo > hi. */
  void InitEmptySet();

  /** Initializes to -infinity to infinity. */
  void InitUniversalSet();
  
  /** Initializes to a range of values. */
  void Init(T lo_in, T hi_in) {
    lo = lo_in;
    hi = hi_in;
  }

  /**
   * Resets to a range of values.
   *
   * Since there is no dynamic memory this is the same as Init, but calling
   * Reset instead of Init probably looks more similar to surrounding code.
   */
  void Reset(T lo_in, T hi_in) {
    lo = lo_in;
    hi = hi_in;
  }

  /**
   * Gets the span of the range, hi - lo.
   */  
  T width() const {
    return hi - lo;
  }

  /**
   * Gets the midpoint of this range.
   */  
  double mid() const {
    return (hi + lo) / 2.0;
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
  const GenRange& operator |= (T d) {
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
  const GenRange& operator &= (T d) {
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
  const GenRange& operator |= (const GenRange& other) {
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
  const GenRange& operator &= (const GenRange& other) {
    if (unlikely(other.lo > lo)) {
      lo = other.lo;
    }
    if (unlikely(other.hi < hi)) {
      hi = other.hi;
    }
    return *this;
  }

  /** Scales upper and lower bounds. */
  friend GenRange operator - (const GenRange& r) {
    return GenRange(-r.hi, -r.lo);
  }
  
  /** Scales upper and lower bounds. */
  const GenRange& operator *= (T d) {
    DEBUG_ASSERT_MSG
      (d >= 0, "don't multiply DRanges by negatives, explicitly negate");
    lo *= d;
    hi *= d;
    return *this;
  }

  /** Scales upper and lower bounds. */
  friend GenRange<T> operator * (const GenRange<T>& r, double d) {
    DEBUG_ASSERT_MSG
      (d >= 0, "don't multiply DRanges by negatives, explicitly negate");
    return GenRange(r.lo * d, r.hi * d);
  }

  /** Scales upper and lower bounds. */
  friend GenRange operator * (T d, const GenRange& r) {
    DEBUG_ASSERT_MSG
      (d >= 0, "don't multiply DRanges by negatives, explicitly negate");
    return GenRange(r.lo * d, r.hi * d);
  }
  
  /** Sums the upper and lower independently. */
  const GenRange& operator += (const GenRange& other) {
    lo += other.lo;
    hi += other.hi;
    return *this;
  }
  
  /** Subtracts from the upper and lower.
   * THIS SWAPS THE ORDER OF HI AND LO, assuming a worst case result.
   * This is NOT an undo of the + operator.
   */
  const GenRange& operator -= (const GenRange& other) {
    lo -= other.hi;
    hi -= other.lo;
    return *this;
  }
  
  /** Adds to the upper and lower independently. */
  const GenRange& operator += (T d) {
    lo += d;
    hi += d;
    return *this;
  }
  
  /** Subtracts from the upper and lower independently. */
  const GenRange& operator -= (T d) {
    lo -= d;
    hi -= d;
    return *this;
  }

  friend GenRange operator + (const GenRange& a, const GenRange& b) {
    GenRange result;
    result.lo = a.lo + b.lo;
    result.hi = a.hi + b.hi;
    return result;
  }

  friend GenRange operator - (const GenRange& a, const GenRange& b) {
    GenRange result;
    result.lo = a.lo - b.hi;
    result.hi = a.hi - b.lo;
    return result;
  }
  
  friend GenRange operator + (const GenRange& a, T b) {
    GenRange result;
    result.lo = a.lo + b;
    result.hi = a.hi + b;
    return result;
  }

  friend GenRange operator - (const GenRange& a, T b) {
    GenRange result;
    result.lo = a.lo - b;
    result.hi = a.hi - b;
    return result;
  }

  /**
   * Takes the maximum of upper and lower bounds independently.
   */
  void MaxWith(const GenRange& range) {
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
  void MinWith(const GenRange& range) {
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
  void MaxWith(T v) {
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
  void MinWith(T v) {
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
  friend bool operator < (const GenRange& a, const GenRange& b) {
    return a.hi < b.lo;
  }
  EXPAND_LESS_THAN(GenRange);
  /**
   * Compares if this is STRICTLY equal to another range.
   */  
  friend bool operator == (const GenRange& a, const GenRange& b) {
    return a.lo == b.lo && a.hi == b.hi;
  }
  EXPAND_EQUALS(GenRange);
  
  /**
   * Compares if this is STRICTLY less than a value.
   */  
  friend bool operator < (const GenRange& a, T b) {
    return a.hi < b;
  }
  EXPAND_HETERO_LESS_THAN(GenRange, T);
  /**
   * Compares if a value is STRICTLY less than this range.
   */  
  friend bool operator < (T a, const GenRange& b) {
    return a < b.lo;
  }
  EXPAND_HETERO_LESS_THAN(T, GenRange);

  /**
   * Determines if a point is contained within the range.
   */  
  bool Contains(T d) const {
    return d >= lo || d <= hi;
  }
};

#endif
