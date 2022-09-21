/**
 * @file core/math/range_impl.hpp
 *
 * Implementation of the (inlined) Range class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_RANGE_IMPL_HPP
#define MLPACK_CORE_MATH_RANGE_IMPL_HPP

#include "range.hpp"
#include <float.h>
#include <sstream>

namespace mlpack {

/**
 * Initialize the range to 0.
 */
template<typename T>
inline RangeType<T>::RangeType() :
    lo(std::numeric_limits<T>::max()),
    hi(-std::numeric_limits<T>::max()) { /* nothing else to do */ }

/**
 * Initialize a range to enclose only the given point.
 */
template<typename T>
inline RangeType<T>::RangeType(const T point) :
    lo(point), hi(point) { /* nothing else to do */ }

/**
 * Initializes the range to the specified values.
 */
template<typename T>
inline RangeType<T>::RangeType(const T lo, const T hi) :
    lo(lo), hi(hi) { /* nothing else to do */ }

/**
 * Gets the span of the range, hi - lo.  Returns 0 if the range is negative.
 */
template<typename T>
inline T RangeType<T>::Width() const
{
  if (lo < hi)
    return (hi - lo);
  else
    return 0.0;
}

/**
 * Gets the midpoint of this range.
 */
template<typename T>
inline T RangeType<T>::Mid() const
{
  return (hi + lo) / 2;
}

/**
 * Expands range to include the other range.
 */
template<typename T>
inline RangeType<T>& RangeType<T>::operator|=(const RangeType<T>& rhs)
{
  if (rhs.lo < lo)
    lo = rhs.lo;
  if (rhs.hi > hi)
    hi = rhs.hi;

  return *this;
}

template<typename T>
inline RangeType<T> RangeType<T>::operator|(const RangeType<T>& rhs) const
{
  return RangeType<T>((rhs.lo < lo) ? rhs.lo : lo,
                      (rhs.hi > hi) ? rhs.hi : hi);
}

/**
 * Shrinks range to be the overlap with another range, becoming an empty
 * set if there is no overlap.
 */
template<typename T>
inline RangeType<T>& RangeType<T>::operator&=(const RangeType<T>& rhs)
{
  if (rhs.lo > lo)
    lo = rhs.lo;
  if (rhs.hi < hi)
    hi = rhs.hi;

  return *this;
}

template<typename T>
inline RangeType<T> RangeType<T>::operator&(const RangeType<T>& rhs) const
{
  return RangeType<T>((rhs.lo > lo) ? rhs.lo : lo,
                      (rhs.hi < hi) ? rhs.hi : hi);
}

/**
 * Scale the bounds by the given double.
 */
template<typename T>
inline RangeType<T>& RangeType<T>::operator*=(const T d)
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

template<typename T>
inline RangeType<T> RangeType<T>::operator*(const T d) const
{
  double nlo = lo * d;
  double nhi = hi * d;

  if (nlo <= nhi)
    return RangeType<T>(nlo, nhi);
  else
    return RangeType<T>(nhi, nlo);
}

// Symmetric case.
template<typename T>
inline RangeType<T> operator*(const T d, const RangeType<T>& r)
{
  double nlo = r.lo * d;
  double nhi = r.hi * d;

  if (nlo <= nhi)
    return RangeType<T>(nlo, nhi);
  else
    return RangeType<T>(nhi, nlo);
}

/**
 * Compare with another range for strict equality.
 */
template<typename T>
inline bool RangeType<T>::operator==(const RangeType<T>& rhs) const
{
  return (lo == rhs.lo) && (hi == rhs.hi);
}

template<typename T>
inline bool RangeType<T>::operator!=(const RangeType<T>& rhs) const
{
  return (lo != rhs.lo) || (hi != rhs.hi);
}

/**
 * Compare with another range.  For Range objects x and y, x < y means that x is
 * strictly less than y and does not overlap at all.
 */
template<typename T>
inline bool RangeType<T>::operator<(const RangeType<T>& rhs) const
{
  return hi < rhs.lo;
}

template<typename T>
inline bool RangeType<T>::operator>(const RangeType<T>& rhs) const
{
  return lo > rhs.hi;
}

/**
 * Determines if a point is contained within the range.
 */
template<typename T>
inline bool RangeType<T>::Contains(const T d) const
{
  return d >= lo && d <= hi;
}

/**
 * Determines if this range overlaps with another range.
 */
template<typename T>
inline bool RangeType<T>::Contains(const RangeType<T>& r) const
{
  return lo <= r.hi && hi >= r.lo;
}

//! Serialize the range.
template<typename T>
template<typename Archive>
void RangeType<T>::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(hi));
  ar(CEREAL_NVP(lo));
}

} // namespace mlpack

#endif
