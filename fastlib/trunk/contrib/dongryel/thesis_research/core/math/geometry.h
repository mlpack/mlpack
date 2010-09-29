/**
 * @file geometry.h
 *
 * Geometric utilities.
 */

#ifndef CORE_MATH_GEOMETRY_H
#define CORE_MATH_GEOMETRY_H

#include <math.h>

namespace core {
namespace math {
/**
 * Computes the hyper-volume of a hyper-sphere of dimension d.
 *
 * @param r the radius of the hyper-sphere
 * @param d the number of dimensions
 */
__attribute__((const)) double SphereVolume(double r, int d);
};
};

#endif
