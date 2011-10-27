/**
 * @file range.hpp
 *
 * Definition of the Range class, which represents a simple range with a lower
 * and upper bound.
 */
#ifndef __MLPACK_CORE_MATH_RANGE_HPP
#define __MLPACK_CORE_MATH_RANGE_HPP

namespace mlpack {
namespace math {

/**
 * Simple real-valued range.  It contains an upper and lower bound.
 */
class Range
{
 public:
  double lo; /// The lower bound.
  double hi; /// The upper bound.

  /** Initialize to an empty set (where lo > hi). */
  Range();

  /***
   * Initialize a range to enclose only the given point (lo = point, hi =
   * point).
   *
   * @param point Point that this range will enclose.
   */
  Range(double point);

  /**
   * Initializes to specified range.
   *
   * @param lo_in Lower bound of the range.
   * @param hi_in Upper bound of the range.
   */
  Range(double lo_in, double hi_in);

  /**
   * Gets the span of the range (hi - lo).
   */
  double width() const;

  /**
   * Gets the midpoint of this range.
   */
  double mid() const;

  /**
   * Expands this range to include another range.
   *
   * @param rhs Range to include.
   */
  Range& operator|=(const Range& rhs);

  /**
   * Expands this range to include another range.
   *
   * @param rhs Range to include.
   */
  Range operator|(const Range& rhs) const;

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   *
   * @param rhs Other range.
   */
  Range& operator&=(const Range& rhs);

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   *
   * @param rhs Other range.
   */
  Range operator&(const Range& rhs) const;

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  Range& operator*=(const double d);

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  Range operator*(const double d) const;

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  friend Range operator*(const double d, const Range& r); // Symmetric case.

  /**
   * Compare with another range for strict equality.
   *
   * @param rhs Other range.
   */
  bool operator==(const Range& rhs) const;

  /**
   * Compare with another range for strict equality.
   *
   * @param rhs Other range.
   */
  bool operator!=(const Range& rhs) const;

  /**
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   *
   * @param rhs Other range.
   */
  bool operator<(const Range& rhs) const;

  /**
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   *
   * @param rhs Other range.
   */
  bool operator>(const Range& rhs) const;

  /**
   * Determines if a point is contained within the range.
   *
   * @param d Point to check.
   */
  bool Contains(double d) const;
};

}; // namespace math
}; // namespace mlpack

#endif // __MLPACK_CORE_MATH_RANGE_HPP
