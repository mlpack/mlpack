/**
 * @file core/tree/hrectbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the HRectBound class, which implements
 * a hyperrectangle bound.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_HRECTBOUND_HPP
#define MLPACK_CORE_TREE_HRECTBOUND_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/range.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include "bound_traits.hpp"

namespace mlpack {

//! Utility struct where Value is true if and only if the argument is of type
//! LMetric.
template<typename DistanceType>
struct IsLMetric
{
  static const bool Value = false;
};

//! Specialization for IsLMetric when the argument is of type LMetric.
template<int Power, bool TakeRoot>
struct IsLMetric<LMetric<Power, TakeRoot>>
{
  static const bool Value = true;
};

/**
 * Hyper-rectangle bound for an L-metric.  This should be used in conjunction
 * with the LMetric class.  Be sure to use the same template parameters for
 * LMetric as you do for HRectBound -- otherwise odd results may occur.
 *
 * @tparam DistanceType Type of distance metric to use; must be of type LMetric.
 * @tparam ElemType Element type (double/float/int/etc.).
 */
template<typename DistanceType = LMetric<2, true>,
         typename ElemType = double>
class HRectBound
{
  // It is required that HRectBound have an LMetric as the given DistanceType.
  static_assert(IsLMetric<DistanceType>::Value == true,
      "HRectBound can only be used with the LMetric<> metric type.");

 public:
  /**
   * Empty constructor; creates a bound of dimensionality 0.
   */
  HRectBound();

  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   *
   * @param dimension Dimensionality of bound.
   */
  HRectBound(const size_t dimension);

  //! Copy constructor; necessary to prevent memory leaks.
  HRectBound(const HRectBound& other);

  //! Same as copy constructor; necessary to prevent memory leaks.
  HRectBound& operator=(const HRectBound& other);

  //! Move constructor: take possession of another bound's information.
  HRectBound(HRectBound&& other);

  //! Move assignment operator.
  HRectBound& operator=(HRectBound&& other);

  //! Destructor: clean up memory.
  ~HRectBound();

  /**
   * Resets all dimensions to the empty set (so that this bound contains
   * nothing).
   */
  void Clear();

  //! Gets the dimensionality.
  size_t Dim() const { return dim; }

  //! Get the range for a particular dimension.  No bounds checking.  Be
  //! careful: this may make MinWidth() invalid.
  RangeType<ElemType>& operator[](const size_t i) { return bounds[i]; }
  //! Modify the range for a particular dimension.  No bounds checking.
  const RangeType<ElemType>& operator[](const size_t i) const
  { return bounds[i]; }

  //! Get the minimum width of the bound.
  ElemType MinWidth() const { return minWidth; }
  //! Modify the minimum width of the bound.
  ElemType& MinWidth() { return minWidth; }

  //! Recompute the minimum width of the bound.
  void RecomputeMinWidth();

  //! Get the instantiated distance metric associated with the bound.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  const DistanceType& Metric() const { return distance; }
  //! Modify the instantiated distance metric associated with the bound.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  DistanceType& Metric() { return distance; }

  //! Get the instantiated distance metric associated with the bound.
  const DistanceType& Distance() const { return distance; }
  //! Modify the instantiated distance metric associated with the bound.
  DistanceType& Distance() { return distance; }

  /**
   * Calculates the center of the range, placing it into the given vector.
   *
   * @param center Vector which the center will be written to.
   */
  void Center(arma::Col<ElemType>& center) const;

  /**
   * Calculate the volume of the hyperrectangle.
   *
   * @return Volume of the hyperrectangle.
   */
  ElemType Volume() const;

  /**
   * Calculates minimum bound-to-point distance.
   *
   * @param point Point to which the minimum distance is requested.
   */
  template<typename VecType>
  ElemType MinDistance(const VecType& point,
                       typename std::enable_if_t<IsVector<VecType>::value>* = 0)
      const;

  /**
   * Calculates minimum bound-to-bound distance.
   *
   * @param other Bound to which the minimum distance is requested.
   */
  ElemType MinDistance(const HRectBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   *
   * @param point Point to which the maximum distance is requested.
   */
  template<typename VecType>
  ElemType MaxDistance(const VecType& point,
                       typename std::enable_if_t<IsVector<VecType>::value>* = 0)
      const;

  /**
   * Computes maximum distance.
   *
   * @param other Bound to which the maximum distance is requested.
   */
  ElemType MaxDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * @param other Bound to which the minimum and maximum distances are
   *     requested.
   */
  RangeType<ElemType> RangeDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   *
   * @param point Point to which the minimum and maximum distances are
   *     requested.
   */
  template<typename VecType>
  RangeType<ElemType> RangeDistance(
      const VecType& point,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  /**
   * Expands this region to include new points.
   *
   * @tparam MatType Type of matrix; could be Mat, SpMat, a subview, or just a
   *   vector.
   * @param data Data points to expand this region to include.
   */
  template<typename MatType>
  HRectBound& operator|=(const MatType& data);

  /**
   * Expands this region to encompass another bound.
   */
  HRectBound& operator|=(const HRectBound& other);

  /**
   * Determines if a point is within this bound.
   *
   * @param point Point to check the condition.
   */
  template<typename VecType>
  bool Contains(const VecType& point) const;

  /**
   * Determines if this bound partially contains a bound.
   *
   * @param bound Bound to check the condition.
   */
  bool Contains(const HRectBound& bound) const;

  /**
   * Returns the intersection of this bound and another.
   */
  HRectBound operator&(const HRectBound& bound) const;

  /**
   * Intersects this bound with another.
   */
  HRectBound& operator&=(const HRectBound& bound);

  /**
   * Returns the volume of overlap of this bound and another.
   */
  ElemType Overlap(const HRectBound& bound) const;

  /**
   * Returns the diameter of the hyperrectangle (that is, the longest diagonal).
   */
  ElemType Diameter() const;

  /**
   * Serialize the bound object.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);

 private:
  //! The dimensionality of the bound.
  size_t dim;
  //! The bounds for each dimension.
  RangeType<ElemType>* bounds;
  //! Cached minimum width of bound.
  ElemType minWidth;
  //! Instantiated distance metric (likely has size 0).
  DistanceType distance;
};

// A specialization of BoundTraits for this class.
template<typename DistanceType, typename ElemType>
struct BoundTraits<HRectBound<DistanceType, ElemType>>
{
  //! These bounds are always tight for each dimension.
  static const bool HasTightBounds = true;
};

} // namespace mlpack

#include "hrectbound_impl.hpp"

#endif // MLPACK_CORE_TREE_HRECTBOUND_HPP
