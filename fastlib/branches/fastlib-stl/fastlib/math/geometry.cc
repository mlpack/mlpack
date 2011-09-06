/**
 * @file geometry.cc
 *
 * Implementation for geometry helpers.
 */

#include "geometry.h"
#include "discrete.h"
#include "math_lib.h"
#include "fastlib/fx/io.h"
//#include "geometry.h"
//#include "discrete.h"
//#include "math_lib.h"

namespace math {

  double SphereVolume(double r, int d) {
    int n = d / 2;
    double val;
    
    mlpack::IO::Assert(d >= 0);
    
    if (d % 2 == 0) {
      val = pow(r * sqrt(PI), d) / Factorial(n);
    }
    else {
      val = pow(2 * r, d) * pow(PI, n) * Factorial(n) / Factorial(d);
    }
    
    return val;
  }
  
};
