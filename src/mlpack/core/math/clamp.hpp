/**
 * @file clamp.hpp
 *
 * Miscellaneous math clamping routines.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_MATH_CLAMP_HPP
#define __MLPACK_CORE_MATH_CLAMP_HPP

#include <stdlib.h>
#include <math.h>
#include <float.h>

namespace mlpack {
namespace math /** Miscellaneous math routines. */ {

/**
 * Forces a number to be non-negative, turning negative numbers into zero.
 * Avoids branching costs (this is a measurable improvement).
 *
 * @param d Double to clamp.
 * @return 0 if d < 0, d otherwise.
 */
inline double ClampNonNegative(const double d)
{
  return (d + fabs(d)) / 2;
}

/**
 * Forces a number to be non-positive, turning positive numbers into zero.
 * Avoids branching costs (this is a measurable improvement).
 *
 * @param d Double to clamp.
 * @param 0 if d > 0, d otherwise.
 */
inline double ClampNonPositive(const double d)
{
  return (d - fabs(d)) / 2;
}

/**
 * Clamp a number between a particular range.
 *
 * @param value The number to clamp.
 * @param rangeMin The first of the range.
 * @param rangeMax The last of the range.
 * @return max(rangeMin, min(rangeMax, d)).
 */
inline double ClampRange(double value,
                         const double rangeMin,
                         const double rangeMax)
{
  value -= rangeMax;
  value = ClampNonPositive(value) + rangeMax;
  value -= rangeMin;
  value = ClampNonNegative(value) + rangeMin;
  return value;
}

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_CLAMP_HPP
