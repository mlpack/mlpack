/**
 * @file range_impl.hpp
 *
 * Implementation of the (inlined) Range class.
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
#ifndef __MLPACK_CORE_MATH_RANGE_IMPL_HPP
#define __MLPACK_CORE_MATH_RANGE_IMPL_HPP

#include "range.hpp"
#include <float.h>
#include <sstream>

namespace mlpack {
namespace math {

/**
 * Initialize the range to 0.
 */
inline Range::Range() :
    lo(DBL_MAX), hi(-DBL_MAX) { /* nothing else to do */ }

/**
 * Initialize a range to enclose only the given point.
 */
inline Range::Range(const double point) :
    lo(point), hi(point) { /* nothing else to do */ }

/**
 * Initializes the range to the specified values.
 */
inline Range::Range(const double lo, const double hi) :
    lo(lo), hi(hi) { /* nothing else to do */ }

/**
 * Gets the span of the range, hi - lo.  Returns 0 if the range is negative.
 */
inline double Range::Width() const
{
  if (lo < hi)
    return (hi - lo);
  else
    return 0.0;
}

/**
 * Gets the midpoint of this range.
 */
inline double Range::Mid() const
{
  return (hi + lo) / 2;
}

/**
 * Expands range to include the other range.
 */
inline Range& Range::operator|=(const Range& rhs)
{
  if (rhs.lo < lo)
    lo = rhs.lo;
  if (rhs.hi > hi)
    hi = rhs.hi;

  return *this;
}

inline Range Range::operator|(const Range& rhs) const
{
  return Range((rhs.lo < lo) ? rhs.lo : lo,
               (rhs.hi > hi) ? rhs.hi : hi);
}

/**
 * Shrinks range to be the overlap with another range, becoming an empty
 * set if there is no overlap.
 */
inline Range& Range::operator&=(const Range& rhs)
{
  if (rhs.lo > lo)
    lo = rhs.lo;
  if (rhs.hi < hi)
    hi = rhs.hi;

  return *this;
}

inline Range Range::operator&(const Range& rhs) const
{
  return Range((rhs.lo > lo) ? rhs.lo : lo,
               (rhs.hi < hi) ? rhs.hi : hi);
}

/**
 * Scale the bounds by the given double.
 */
inline Range& Range::operator*=(const double d)
{
  lo *= d;
  hi *= d;

  // Now if we've negated, we need to flip things around so the bound is valid.
  if (lo > hi)
  {
    double tmp = hi;
    hi = lo;
    lo = tmp;
  }

  return *this;
}

inline Range Range::operator*(const double d) const
{
  double nlo = lo * d;
  double nhi = hi * d;

  if (nlo <= nhi)
    return Range(nlo, nhi);
  else
    return Range(nhi, nlo);
}

// Symmetric case.
inline Range operator*(const double d, const Range& r)
{
  double nlo = r.lo * d;
  double nhi = r.hi * d;

  if (nlo <= nhi)
    return Range(nlo, nhi);
  else
    return Range(nhi, nlo);
}

/**
 * Compare with another range for strict equality.
 */
inline bool Range::operator==(const Range& rhs) const
{
  return (lo == rhs.lo) && (hi == rhs.hi);
}

inline bool Range::operator!=(const Range& rhs) const
{
  return (lo != rhs.lo) || (hi != rhs.hi);
}

/**
 * Compare with another range.  For Range objects x and y, x < y means that x is
 * strictly less than y and does not overlap at all.
 */
inline bool Range::operator<(const Range& rhs) const
{
  return hi < rhs.lo;
}

inline bool Range::operator>(const Range& rhs) const
{
  return lo > rhs.hi;
}

/**
 * Determines if a point is contained within the range.
 */
inline bool Range::Contains(const double d) const
{
  return d >= lo && d <= hi;
}

/**
 * Determines if this range overlaps with another range.
 */
inline bool Range::Contains(const Range& r) const
{
  return lo <= r.hi && hi >= r.lo;
}

/**
 * Returns a string representation of an object.
 */
std::string Range::ToString() const
{
  std::ostringstream convert;
  convert << "[" << lo << ", " << hi << "]";
  return convert.str();
}
}; // namespace math
}; // namespace mlpack

#endif
