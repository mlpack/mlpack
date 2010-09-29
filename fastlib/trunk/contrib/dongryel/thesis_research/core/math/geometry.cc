/**
 * @file geometry.cc
 *
 * Implementation for geometry helpers.
 */

#include "geometry.h"
#include "math_lib.h"
#include "boost/math/constants/constants.hpp"
#include "boost/math/special_functions/factorials.hpp"

namespace core {
namespace math {
double SphereVolume(double r, int d) {
  int n = d / 2;
  double val;

  if(d % 2 == 0) {
    val = pow(r * boost::math::constants::root_pi<double>(), d) /
          boost::math::factorial<double>(n);
  }
  else {
    val = pow(2 * r, d) * pow(boost::math::constants::pi<double>(), n) *
          boost::math::factorial<double>(n) / boost::math::factorial<double>(d);
  }
  return val;
}
};
};
