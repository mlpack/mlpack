// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file math.h
 *
 * Includes all basic FASTlib non-vector math utilities.
 */

#ifndef MATH_MATH_H
#define MATH_MATH_H

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
  /** The ratio of the radius of a circle to its diameter. */
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
};

#include "discrete.h"
#include "kernel.h"
#include "geometry.h"

#endif
