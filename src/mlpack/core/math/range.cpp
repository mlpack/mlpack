/**
 * @file range.cpp
 *
 * Implementation of the Range class.
 */
#include "range.hpp"
#include <float.h>

namespace mlpack {
namespace math {

/**
 * Initialize the range to 0.
 */
Range::Range() :
    lo(DBL_MAX), hi(-DBL_MAX) { /* nothing else to do */ }

/**
 * Initialize a range to enclose only the given point.
 */
Range::Range(double point) :
    lo(point), hi(point) { /* nothing else to do */ }

/**
 * Initializes the range to the specified values.
 */
Range::Range(double lo_in, double hi_in) :
    lo(lo_in), hi(hi_in) { /* nothing else to do */ }

/**
 * Gets the span of the range, hi - lo.  Returns 0 if the range is negative.
 */
double Range::width() const
{
  if (lo < hi)
    return (hi - lo);
  else
    return 0.0;
}

/**
 * Gets the midpoint of this range.
 */
double Range::mid() const
{
  return (hi + lo) / 2;
}

/**
 * Expands range to include the other range.
 */
Range& Range::operator|=(const Range& rhs)
{
  if (rhs.lo < lo)
    lo = rhs.lo;
  if (rhs.hi > hi)
    hi = rhs.hi;

  return *this;
}

Range Range::operator|(const Range& rhs) const
{
  return Range((rhs.lo < lo) ? rhs.lo : lo,
               (rhs.hi > hi) ? rhs.hi : hi);
}

/**
 * Shrinks range to be the overlap with another range, becoming an empty
 * set if there is no overlap.
 */
Range& Range::operator&=(const Range& rhs)
{
  if (rhs.lo > lo)
    lo = rhs.lo;
  if (rhs.hi < hi)
    hi = rhs.hi;

  return *this;
}

Range Range::operator&(const Range& rhs) const
{
  return Range((rhs.lo > lo) ? rhs.lo : lo,
               (rhs.hi < hi) ? rhs.hi : hi);
}

/**
 * Scale the bounds by the given double.
 */
Range& Range::operator*=(const double d)
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

Range Range::operator*(const double d) const
{
  double nlo = lo * d;
  double nhi = hi * d;

  if (nlo <= nhi)
    return Range(nlo, nhi);
  else
    return Range(nhi, nlo);
}

// Symmetric case.
Range operator*(const double d, const Range& r)
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
bool Range::operator==(const Range& rhs) const
{
  return (lo == rhs.lo) && (hi == rhs.hi);
}

bool Range::operator!=(const Range& rhs) const
{
  return (lo != rhs.lo) || (hi != rhs.hi);
}

/**
 * Compare with another range.  For Range objects x and y, x < y means that x is
 * strictly less than y and does not overlap at all.
 */
bool Range::operator<(const Range& rhs) const
{
  return hi < rhs.lo;
}

bool Range::operator>(const Range& rhs) const
{
  return lo > rhs.hi;
}

/**
 * Determines if a point is contained within the range.
 */
bool Range::Contains(double d) const
{
  return d >= lo && d <= hi;
}

}; // namesapce math
}; // namespace mlpack
