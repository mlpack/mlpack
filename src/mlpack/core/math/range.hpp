/**
 * @file core/math/range.hpp
 *
 * Definition of the Range class, which represents a simple range with a lower
 * and upper bound.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_RANGE_HPP
#define MLPACK_CORE_MATH_RANGE_HPP

namespace mlpack {

template<typename T>
class RangeType;

using Range = RangeType<double>;

/**
 * Simple real-valued range.  It contains an upper and lower bound.
 *
 * @tparam T type of element held by this range.
 */
template<typename T = double>
class RangeType
{
 private:
  T lo; /// The lower bound.
  T hi; /// The upper bound.

 public:
  /** Initialize to an empty set (where lo > hi). */
  inline RangeType();

  /***
   * Initialize a range to enclose only the given point (lo = point, hi =
   * point).
   *
   * @param point Point that this range will enclose.
   */
  inline RangeType(const T point);

  /**
   * Initializes to specified range.
   *
   * @param lo Lower bound of the range.
   * @param hi Upper bound of the range.
   */
  inline RangeType(const T lo, const T hi);

  //! Get the lower bound.
  inline T Lo() const { return lo; }
  //! Modify the lower bound.
  inline T& Lo() { return lo; }

  //! Get the upper bound.
  inline T Hi() const { return hi; }
  //! Modify the upper bound.
  inline T& Hi() { return hi; }

  /**
   * Gets the span of the range (hi - lo).
   */
  inline T Width() const;

  /**
   * Gets the midpoint of this range.
   */
  inline T Mid() const;

  /**
   * Expands this range to include another range.
   *
   * @param rhs Range to include.
   */
  inline RangeType& operator|=(const RangeType& rhs);

  /**
   * Expands this range to include another range.
   *
   * @param rhs Range to include.
   */
  inline RangeType operator|(const RangeType& rhs) const;

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   *
   * @param rhs Other range.
   */
  inline RangeType& operator&=(const RangeType& rhs);

  /**
   * Shrinks this range to be the overlap with another range; this makes an
   * empty set if there is no overlap.
   *
   * @param rhs Other range.
   */
  inline RangeType operator&(const RangeType& rhs) const;

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  inline RangeType& operator*=(const T d);

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   */
  inline RangeType operator*(const T d) const;

  /**
   * Scale the bounds by the given double.
   *
   * @param d Scaling factor.
   * @param r Bounds range.
   */
  template<typename TT>
  friend inline RangeType<TT> operator*(const TT d, const RangeType<TT>& r);

  /**
   * Compare with another range for strict equality.
   *
   * @param rhs Other range.
   */
  inline bool operator==(const RangeType& rhs) const;

  /**
   * Compare with another range for strict equality.
   *
   * @param rhs Other range.
   */
  inline bool operator!=(const RangeType& rhs) const;

  /**
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   *
   * @param rhs Other range.
   */
  inline bool operator<(const RangeType& rhs) const;

  /**
   * Compare with another range.  For Range objects x and y, x < y means that x
   * is strictly less than y and does not overlap at all.
   *
   * @param rhs Other range.
   */
  inline bool operator>(const RangeType& rhs) const;

  /**
   * Determines if a point is contained within the range.
   *
   * @param d Point to check.
   */
  inline bool Contains(const T d) const;

  /**
   * Determines if another range overlaps with this one.
   *
   * @param r Other range.
   *
   * @return true if ranges overlap at all.
   */
  inline bool Contains(const RangeType& r) const;

  /**
   * Serialize the range object.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);
};

} // namespace mlpack

// Include inlined implementation.
#include "range_impl.hpp"

#endif // MLPACK_CORE_MATH_RANGE_HPP
