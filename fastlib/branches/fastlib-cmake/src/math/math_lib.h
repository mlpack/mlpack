/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file math_lib.h
 *
 * Includes all basic FASTlib non-vector math utilities.
 */

#ifndef MATH_MATH_LIB_H
#define MATH_MATH_LIB_H

#include "base/base.h"

#include <math.h>

/**
 * Math routines.
 *
 * The hope is that this should contain most of the useful math routines
 * you can think of.  Currently, this is very sparse.
 */
namespace math {
  /** The square root of 2. */
  const double SQRT2 = 1.41421356237309504880;
  /** Base of the natural logarithm. */
  const double E = 2.7182818284590452354;
  /** Log base 2 of E. */
  const double LOG2_E = 1.4426950408889634074;
  /** Log base 10 of E. */
  const double LOG10_E = 0.43429448190325182765;
  /** Natural log of 2. */
  const double LN_2 = 0.69314718055994530942;
  /** Natural log of 10. */
  const double LN_10 = 2.30258509299404568402;
  /** The ratio of the circumference of a circle to its diameter. */
  const double PI = 3.141592653589793238462643383279;
  /** The ratio of the circumference of a circle to its radius. */
  const double PI_2 = 1.57079632679489661923;

  /** Squares a number. */
  template<typename T>
  inline T Sqr(T v) {
    return v * v;
  }

  /**
   * Rounds a double-precision to an integer, casting it too.
   */
  inline int64 RoundInt(double d) {
    return int64(nearbyint(d));
  }

  /**
   * Forces a number to be non-negative, turning negative numbers into zero.
   *
   * Avoids branching costs (yes, we've discovered measurable improvements).
   */
  inline double ClampNonNegative(double d) {
    return (d + fabs(d)) / 2;
  }

  /**
   * Forces a number to be non-positive, turning positive numbers into zero.
   *
   * Avoids branching costs (yes, we've discovered measurable improvements).
   */
  inline double ClampNonPositive(double d) {
    return (d - fabs(d)) / 2;
  }
  
  /**
   * Clips a number between a particular range.
   *
   * @param value the number to clip
   * @param range_min the first of the range
   * @param range_max the last of the range
   * @return max(range_min, min(range_max, d))
   */
  inline double ClampRange(double value, double range_min, double range_max) {
    if (unlikely(value <= range_min)) {
      return range_min;
    } else if (unlikely(value >= range_max)) {
      return range_max; 
    } else {
      return value;
    }
  }
  
  /**
   * Generates a uniform random number between 0 and 1.
   */
  inline double Random() {
    return rand() * (1.0 / RAND_MAX);
  }
  
  /**
   * Generates a uniform random number in the specified range.
   */
  inline double Random(double lo, double hi) {
    return Random() * (hi - lo) + lo;
  }

  /**
   * Generates a uniform random integer.
   */
  inline int RandInt(int hi_exclusive) {
    return rand() % hi_exclusive;
  }  
  /**
   * Generates a uniform random integer.
   */
  inline int RandInt(int lo, int hi_exclusive) {
    return (rand() % (hi_exclusive - lo)) + lo;
  }
};

#include "math_lib_impl.h"
//#include "math_lib_impl.h"

namespace math {
  /**
   * Calculates a relatively small power using template metaprogramming.
   *
   * This allows a numerator and denominator.  In the case where the
   * numerator and denominator are equal, this will not do anything, or in
   * the case where the denominator is one.
   */
  template<int t_numerator, int t_denominator> 
  inline double Pow(double d) {
    return math__private::ZPowImpl<t_numerator, t_denominator>::Calculate(d);
  }
  
  /**
   * Calculates a small power of the absolute value of a number
   * using template metaprogramming.
   *
   * This allows a numerator and denominator.  In the case where the
   * numerator and denominator are equal, this will not do anything, or in
   * the case where the denominator is one.  For even powers, this will
   * avoid calling the absolute value function.
   */
  template<int t_numerator, int t_denominator> 
  inline double PowAbs(double d) {
    // we specify whether it's an even function -- if so, we can sometimes
    // avoid the absolute value sign
    return math__private::ZPowAbsImpl<t_numerator, t_denominator,
        (t_numerator%t_denominator == 0) && ((t_numerator/t_denominator)%2 == 0)>::Calculate(fabs(d));
  }
};

/**
 * A value which is the min or max of multiple other values.
 *
 * Comes with a highly optimized version of x = max(x, y).
 *
 * The template argument should be something like double, with greater-than,
 * less-than, and equals operators.
 */
template<typename TValue>
class MinMaxVal {
 public:
  typedef TValue Value;

 public:
  /** The underlying value. */
  Value val;

  OBJECT_TRAVERSAL(MinMaxVal) {
    OT_OBJ(val);
  }

 public:
  /**
   * Converts implicitly to the value.
   */
  operator Value() const { return val; }

  /**
   * Sets the value.
   */
  const Value& operator = (Value val_in) {
    return (val = val_in);
  }

  /**
   * Efficiently performs this->val = min(this->val, incoming_val).
   *
   * The expectation is that it is higly unlikely for the incoming
   * value to be the new minimum.
   */
  void MinWith(Value incoming_val) {
    if (unlikely(incoming_val < val)) {
      val = incoming_val;
    }
  }
  
  /**
   * Efficiently performs this->val = min(this->val, incoming_val).
   *
   * The expectation is that it is higly unlikely for the incoming
   * value to be the new maximum.
   */
  void MaxWith(Value incoming_val) {
    if (unlikely(incoming_val > val)) {
      val = incoming_val;
    }
  }
};

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

  OBJECT_TRAVERSAL(DRange) {
    OT_OBJ(lo);
    OT_OBJ(hi);
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

  /** Scales upper and lower bounds. */
  friend DRange operator - (const DRange& r) {
    return DRange(-r.hi, -r.lo);
  }
  
  /** Scales upper and lower bounds. */
  const DRange& operator *= (double d) {
    DEBUG_ASSERT_MSG(d >= 0, "don't multiply DRanges by negatives, explicitly negate");
    lo *= d;
    hi *= d;
    return *this;
  }

  /** Scales upper and lower bounds. */
  friend DRange operator * (const DRange& r, double d) {
    DEBUG_ASSERT_MSG(d >= 0, "don't multiply DRanges by negatives, explicitly negate");
    return DRange(r.lo * d, r.hi * d);
  }

  /** Scales upper and lower bounds. */
  friend DRange operator * (double d, const DRange& r) {
    DEBUG_ASSERT_MSG(d >= 0, "don't multiply DRanges by negatives, explicitly negate");
    return DRange(r.lo * d, r.hi * d);
  }
  
  /** Sums the upper and lower independently. */
  const DRange& operator += (const DRange& other) {
    lo += other.lo;
    hi += other.hi;
    return *this;
  }
  
  /** Subtracts from the upper and lower.
   * THIS SWAPS THE ORDER OF HI AND LO, assuming a worst case result.
   * This is NOT an undo of the + operator.
   */
  const DRange& operator -= (const DRange& other) {
    lo -= other.hi;
    hi -= other.lo;
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
    result.lo = a.lo - b.hi;
    result.hi = a.hi - b.lo;
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
  EXPAND_LESS_THAN(DRange);
  /**
   * Compares if this is STRICTLY equal to another range.
   */  
  friend bool operator == (const DRange& a, const DRange& b) {
    return a.lo == b.lo && a.hi == b.hi;
  }
  EXPAND_EQUALS(DRange);
  
  /**
   * Compares if this is STRICTLY less than a value.
   */  
  friend bool operator < (const DRange& a, double b) {
    return a.hi < b;
  }
  EXPAND_HETERO_LESS_THAN(DRange, double);
  /**
   * Compares if a value is STRICTLY less than this range.
   */  
  friend bool operator < (double a, const DRange& b) {
    return a < b.lo;
  }
  EXPAND_HETERO_LESS_THAN(double, DRange);

  /**
   * Determines if a point is contained within the range.
   */  
  bool Contains(double d) const {
    return d >= lo || d <= hi;
  }
};

#include "discrete.h"
#include "kernel.h"
#include "geometry.h"
//#include "discrete.h"
//#include "kernel.h"
//#include "geometry.h"

#endif
