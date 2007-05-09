// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file math.h
 *
 * Includes all basic FASTlib non-vector math utilities.
 */

#ifndef MATH_MATH_H
#define MATH_MATH_H

#include "base/common.h"
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
};

namespace math_private {
  template<int t_numerator, int t_denominator = 1>
  struct PowImpl {
    static double Calculate(double d) {
      return pow(d, t_numerator * 1.0 / t_denominator);
    }
  };

  template<int t_equal>
  struct PowImpl<t_equal, t_equal> {
    static double Calculate(double d) {
      return d;
    }
  };

  template<>
  struct PowImpl<1, 1> {
    static double Calculate(double d) {
      return d;
    }
  };

  template<>
  struct PowImpl<1, 2> {
    static double Calculate(double d) {
      return sqrt(d);
    }
  };

  template<>
  struct PowImpl<1, 3> {
    static double Calculate(double d) {
      return cbrt(d);
    }
  };

  template<int t_denominator>
  struct PowImpl<0, t_denominator> {
    static double Calculate(double d) {
      return 1;
    }
  };

  template<int t_numerator>
  struct PowImpl<t_numerator, 1> {
    static double Calculate(double d) {
      return PowImpl<t_numerator - 1, 1>::Calculate(d) * d;
    }
  };
};

namespace math {
  /**
   * Calculates a relatively small power using template metaprogramming.
   *
   * This allows a numerator and denominator.  In the case where the numerator
   * and denominator are equal, this will not do anything, or in the case where
   * the denominator is one.
   */
  template<int t_numerator, int t_denominator> 
  inline double Pow(double d) {
    return math_private::PowImpl<t_numerator, t_denominator>::Calculate(d);
  }
  
  /**
   * Calculates a small power of the absolute value of a number
   * using template metaprogramming.
   *
   * This allows a numerator and denominator.  In the case where the numerator
   * and denominator are equal, this will not do anything, or in the case where
   * the denominator is one.
   */
  template<int t_numerator, int t_denominator> 
  inline double PowAbs(double d) {
    return math_private::PowImpl<t_numerator, t_denominator>::Calculate(fabs(d));
  }
};

#include "discrete.h"
#include "kernel.h"
#include "geometry.h"

#endif
