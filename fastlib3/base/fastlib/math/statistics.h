// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file statistics.h
 *
 * Statistics utilities.
 */

#ifndef MATH_STATISTICS_H
#define MATH_STATISTICS_H

#include "fastlib/base/base.h"
#include "fastlib/la/matrix.h"
#include <math.h>

namespace math {
  /**
   * Computes the mean value of a vector.
   * Don't forget initializing V before using this function
   *
   * @param V the input vector
   * @return the mean value
   */
  double Mean(Vector V);

  /**
   * Computes the variance of a vector using "corrected two-pass algorithm".
   * See "Numerical Recipes in C" for reference.
   * Don't forget initializing V before using this function.
   *
   * @param V the input vector
   * @return the variance
   */
  double Var(Vector V);

  /**
   * Computes the standard deviation of a vector.
   * Don't forget initializing V before using this function.
   *
   * @param V the input vector
   * @return the standard deviation
   */
  double Std(Vector V);

  /**
   * Computes the sigmoid function of a real number x
   * Sigmoid(x) = 1/[1+exp(-x)]
   *
   * @param x the input real number
   * @return the sigmoid function value
   */
  double Sigmoid(double x);
};

#endif
