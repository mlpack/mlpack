// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file statistics.h
 *
 * Statistics utilities.
 */

#ifndef MATH_STATISTICS_H
#define MATH_STATISTICS_H

#include "base/base.h"
#include "la/matrix.h"
#include <math.h>

namespace math {
  /**
   * Computes the mean value of a vector. Don't forget initializing V before using.
   *
   * @param V the input vector
   * @return the mean value
   */
  double Mean(Vector V);
};

#endif
