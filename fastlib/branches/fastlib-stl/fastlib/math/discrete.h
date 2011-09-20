/**
 * @file discrete.h
 *
 * Discrete math helpers.
 */

#ifndef MATH_DISCRETE_H
#define MATH_DISCRETE_H

#include "fastlib/fx/io.h"

#include <vector>

#include <math.h>

namespace math {

  /**
   * Computes the binomial coefficient, n choose k for nonnegative integers
   * n and k
   *
   * @param n the first nonnegative integer argument
   * @param k the second nonnegative integer argument
   * @return the binomial coefficient n choose k
   */
  __attribute__((const)) double BinomialCoefficient(int n, int k);

  

};

#endif
