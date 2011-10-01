/**
 * @file range.h
 *
 * Definition of the Range class, which represents a simple range with a lower
 * and upper bound.
 */

#ifndef __MATH_RANGE_H
#define __MATH_RANGE_H

/**
 * Simple real-valued range.
 */
class Range {
 public:
  double lo; /// The lower bound.
  double hi; /// The upper bound.

  /** Initialize to an empty set (where lo > hi). */
  Range();

  /***
   * Initialize a range to enclose only the given point (lo = point, hi =
   * point).
   */
  Range(double point);

  /** Initializes to specified values. */
  Range(double lo_in, double hi_in);

  /**
   * Gets the span of the range, hi - lo.
   */
  double width() const;

  /**
   * Gets the midpoint of this range.
   */
  double mid() const;

  /**
   * Expands this range to include another range.
   */
  Range& operator|=(const Range& rhs);
  Range operator|(const Range& rhs) const;

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   */
  Range& operator&=(const Range& rhs);
  Range operator&(const Range& rhs) const;

  /**
   * Scale the bounds by the given double.
   */
  Range& operator*=(const double d);
  Range operator*(const double d) const;

  friend Range operator*(const double d, const Range& r); // Symmetric case.

  /***
   * Compare with another range for strict equality.
   */
  bool operator==(const Range& rhs) const;
  bool operator!=(const Range& rhs) const;

  /***
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   */
  bool operator<(const Range& rhs) const;
  bool operator>(const Range& rhs) const;

  /**
   * Determines if a point is contained within the range.
   */
  bool Contains(double d) const;
};

#endif
