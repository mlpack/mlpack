/**
 * @file range.cc
 *
 * Implementation of the Range class.
 */
#include "range.h"
#include <float.h>

/** Initialize to 0. */
Range::Range() :
    lo(0), hi(0) { /* nothing else to do */ }

/** Initializes to specified values. */
Range::Range(double lo_in, double hi_in) :
    lo(lo_in), hi(hi_in) { /* nothing else to do */ }

/** Initialize to an empty set, where lo > hi. */
void Range::InitEmptySet() {
  lo = DBL_MAX;
  hi = -DBL_MAX;
}

/** Initializes to -infinity to infinity. */
void Range::InitUniversalSet() {
  lo = -DBL_MAX;
  hi = DBL_MAX;
}

/** Initializes to a range of values. */
void Range::Init(double lo_in, double hi_in) {
  lo = lo_in;
  hi = hi_in;
}

/**
 * Resets to a range of values.
 *
 * Since there is no dynamic memory this is the same as Init, but calling
 * Reset instead of Init probably looks more similar to surrounding code.
 */
void Range::Reset(double lo_in, double hi_in) {
  lo = lo_in;
  hi = hi_in;
}

/**
 * Gets the span of the range, hi - lo.
 */
double Range::width() const {
  return hi - lo;
}

/**
 * Gets the midpoint of this range.
 */
double Range::mid() const {
  return (hi + lo) / 2;
}

/**
 * Interpolates (factor) * hi + (1 - factor) * lo.
 */
double Range::interpolate(double factor) const {
  return factor * width() + lo;
}

/**
 * Simulate a union by growing the range if necessary.
 */
const Range& Range::operator|=(double d) {
  if (d < lo)
    lo = d;
  if (d > hi)
    hi = d;

  return *this;
}

/**
 * Sets this range to include only the specified value, or
 * becomes an empty set if the range does not contain the number.
 */
const Range& Range::operator&=(double d) {
  if (d > lo)
    lo = d;
  if (d < hi)
    hi = d;

  return *this;
}

/**
 * Expands range to include the other range.
 */
const Range& Range::operator|=(const Range& other) {
  if (other.lo < lo)
    lo = other.lo;
  if (other.hi > hi)
    hi = other.hi;

  return *this;
}

/**
 * Shrinks range to be the overlap with another range, becoming an empty
 * set if there is no overlap.
 */
const Range& Range::operator&=(const Range& other) {
  if (other.lo > lo)
    lo = other.lo;
  if (other.hi < hi)
    hi = other.hi;

  return *this;
}

/** Scales upper and lower bounds. */
Range operator-(const Range& r) {
  return Range(-r.hi, -r.lo);
}

/** Scales upper and lower bounds. */
const Range& Range::operator*=(double d) {
//  mlpack::IO::AssertMessage(d >= 0, "don't multiply Ranges by negatives, explicitly negate");
  lo *= d;
  hi *= d;

  return *this;
}

/** Scales upper and lower bounds. */
Range operator*(const Range& r, double d) {
//  mlpack::IO::AssertMessage(d >= 0, "don't multiply Ranges by negatives, explicitly negate");

  return Range(r.lo * d, r.hi * d);
}

/** Scales upper and lower bounds. */
Range operator*(double d, const Range& r) {
  //mlpack::IO::AssertMessage(d >= 0, "don't multiply Ranges by negatives, explicitly negate");
  return Range(r.lo * d, r.hi * d);
}

/** Sums the upper and lower independently. */
const Range& Range::operator+=(const Range& other) {
  lo += other.lo;
  hi += other.hi;

  return *this;
}

/** Subtracts from the upper and lower.
 * THIS SWAPS THE ORDER OF HI AND LO, assuming a worst case result.
 * This is NOT an undo of the + operator.
 */
const Range& Range::operator-=(const Range& other) {
  lo -= other.hi;
  hi -= other.lo;

  return *this;
}

/** Adds to the upper and lower independently. */
const Range& Range::operator+=(double d) {
  lo += d;
  hi += d;

  return *this;
}

/** Subtracts from the upper and lower independently. */
const Range& Range::operator-=(double d) {
  lo -= d;
  hi -= d;

  return *this;
}

Range operator+(const Range& a, const Range& b) {
  Range result(a.lo + b.lo, a.hi + b.hi);
  return result;
}

Range operator-(const Range& a, const Range& b) {
  Range result(a.lo - b.hi, a.hi - b.lo);
  return result;
}

Range operator+(const Range& a, double b) {
  Range result(a.lo + b, a.hi + b);
  return result;
}

Range operator-(const Range& a, double b) {
  Range result(a.lo - b, a.hi - b);
  return result;
}

/**
 * Takes the maximum of upper and lower bounds independently.
 */
void Range::MaxWith(const Range& range) {
  if (range.lo > lo)
    lo = range.lo;
  if (range.hi > hi)
    hi = range.hi;
}

/**
 * Takes the minimum of upper and lower bounds independently.
 */
void Range::MinWith(const Range& range) {
  if (range.lo < lo)
    lo = range.lo;
  if (range.hi < hi)
    hi = range.hi;
}

/**
 * Takes the maximum of upper and lower bounds independently.
 */
void Range::MaxWith(double v) {
  if (v > lo) {
    lo = v;
    if (v > hi)
      hi = v;
  }
}

/**
 * Takes the minimum of upper and lower bounds independently.
 */
void Range::MinWith(double v) {
  if (v < hi) {
    hi = v;
    if (v < lo)
      lo = v;
  }
}

/**
 * Compares if this is STRICTLY less than another range.
 */
bool operator<(const Range& a, const Range& b) {
  return a.hi < b.lo;
}

bool operator>(const Range& b, const Range& a) {
  return a < b;
}

bool operator<=(const Range& b, const Range& a) {
  return !(a < b);
}

bool operator>=(const Range& a, const Range& b) {
  return !(a < b);
}

/**
 * Compares if this is STRICTLY equal to another range.
 */
bool operator==(const Range& a, const Range& b) {
  return a.lo == b.lo && a.hi == b.hi;
}

bool operator!=(const Range& a, const Range& b) {
  return !(a == b);
}

/**
 * Compares if this is STRICTLY less than a value.
 */
bool operator<(const Range& a, double b) {
  return a.hi < b;
}

bool operator>(const double& b, const Range& a) {
  return a < b;
}

bool operator<=(const double& b, const Range& a) {
  return !(a < b);
}

bool operator>=(const Range& a, const double& b) {
  return !(a < b);
}

/**
 * Compares if a value is STRICTLY less than this range.
 */
bool operator<(double a, const Range& b) {
  return a < b.lo;
}

bool operator>(const Range& b, const double& a) {
  return a < b;
}

bool operator<=(const Range& b, const double& a) {
  return !(a < b);
}

bool operator>=(const double& a, const Range& b) {
  return !(a < b);
}

/**
 * Determines if a point is contained within the range.
 */
bool Range::Contains(double d) const {
  return d >= lo && d <= hi;
}
