// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file geometry.cc
 *
 * Implementation for geometry helpers.
 */

#include "geometry.h"
#include "discrete.h"
#include "math.h"

namespace math {

double SphereVolume(double r, int d) {
  int n = d / 2;
  double val;
  
  DEBUG_ASSERT(d >= 0);
  
  if (d % 2 == 0) {
    val = pow(r * sqrt(PI), d)
        / Factorial(n);
  } else {
    val = pow(2 * r, d) * pow(PI, n)
        * Factorial(n) / Factorial(d);
  }
  
  return val;
}

};
