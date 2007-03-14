// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file geometry.h
 *
 * Geometric utilities.
 */

#ifndef MATH_GEOMETRY_H
#define MATH_GEOMETRY_H

#include "base/cc.h"

#include <cmath>

namespace math {
  const double PI = 3.141592653589793238462643383279;
  
  /**
   * Computes the hyper-volume of a hyper-sphere of dimension d.
   *
   * @param r the radius of the hyper-sphere
   * @param d the number of dimensions
   */
  COMPILER_FUNCTIONAL double SphereVolume(double r, int d);
};

#endif
