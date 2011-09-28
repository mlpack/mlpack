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
struct Range {
 public:
  double lo; /// The lower bound.
  double hi; /// The upper bound.

  /** Initialize to 0. */
  Range();

  /** Initializes to specified values. */
  Range(double lo_in, double hi_in);

  /** Initialize to an empty set, where lo > hi. */
  void InitEmptySet();

  /** Initializes to -infinity to infinity. */
  void InitUniversalSet();

  /** Initializes to a range of values. */
  void Init(double lo_in, double hi_in);

  /**
   * Resets to a range of values.
   *
   * Since there is no dynamic memory this is the same as Init, but calling
   * Reset instead of Init probably looks more similar to surrounding code.
   */
  void Reset(double lo_in, double hi_in);

  /**
   * Gets the span of the range, hi - lo.
   */
  double width() const;

  /**
   * Gets the midpoint of this range.
   */
  double mid() const;

  /**
   * Interpolates (factor) * hi + (1 - factor) * lo.
   */
  double interpolate(double factor) const;

  /**
   * Simulate a union by growing the range if necessary.
   */
  const Range& operator|=(double d);

  /**
   * Sets this range to include only the specified value, or
   * becomes an empty set if the range does not contain the number.
   */
  const Range& operator&=(double d);

  /**
   * Expands range to include the other range.
   */
  const Range& operator|=(const Range& other);

  /**
   * Shrinks range to be the overlap with another range, becoming an empty
   * set if there is no overlap.
   */
  const Range& operator&=(const Range& other);

  /** Scales upper and lower bounds. */
  friend Range operator-(const Range& r);

  /** Scales upper and lower bounds. */
  const Range& operator*=(double d);

  /** Scales upper and lower bounds. */
  friend Range operator*(const Range& r, double d);

  /** Scales upper and lower bounds. */
  friend Range operator*(double d, const Range& r);

  /** Sums the upper and lower independently. */
  const Range& operator+=(const Range& other);

  /** Subtracts from the upper and lower.
   * THIS SWAPS THE ORDER OF HI AND LO, assuming a worst case result.
   * This is NOT an undo of the + operator.
   */
  const Range& operator-=(const Range& other);

  /** Adds to the upper and lower independently. */
  const Range& operator+=(double d);

  /** Subtracts from the upper and lower independently. */
  const Range& operator-=(double d);

  friend Range operator+(const Range& a, const Range& b);
  friend Range operator-(const Range& a, const Range& b);
  friend Range operator+(const Range& a, double b);
  friend Range operator-(const Range& a, double b);

  /**
   * Takes the maximum of upper and lower bounds independently.
   */
  void MaxWith(const Range& range);

  /**
   * Takes the minimum of upper and lower bounds independently.
   */
  void MinWith(const Range& range);

  /**
   * Takes the maximum of upper and lower bounds independently.
   */
  void MaxWith(double v);

  /**
   * Takes the minimum of upper and lower bounds independently.
   */
  void MinWith(double v);

  /**
   * Compares if this is STRICTLY less than another range.
   */
  friend bool operator<(const Range& a, const Range& b);

  friend bool operator>(const Range& b, const Range& a);
  friend bool operator<=(const Range& b, const Range& a);
  friend bool operator>=(const Range& a, const Range& b);

  /**
   * Compares if this is STRICTLY equal to another range.
   */
  friend bool operator==(const Range& a, const Range& b);

  friend bool operator!=(const Range& a, const Range& b);

  /**
   * Compares if this is STRICTLY less than a value.
   */
  friend bool operator<(const Range& a, double b);
  friend bool operator>(const double& b, const Range& a);
  friend bool operator<=(const double& b, const Range& a);
  friend bool operator>=(const Range& a, const double& b);

  /**
   * Compares if a value is STRICTLY less than this range.
   */
  friend bool operator<(double a, const Range& b);

  friend bool operator>(const Range& b, const double& a);
  friend bool operator<=(const Range& b, const double& a);
  friend bool operator>=(const double& a, const Range& b);

  /**
   * Determines if a point is contained within the range.
   */
  bool Contains(double d) const;
};

#endif
