/**
 * @file hrectbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the HRectBound class, which implements
 * a hyperrectangle bound.
 */
#ifndef __MLPACK_CORE_TREE_HRECTBOUND_HPP
#define __MLPACK_CORE_TREE_HRECTBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/math/range.hpp>

namespace mlpack {
namespace bound {

/**
 * Hyper-rectangle bound for an L-metric.  This should be used in conjunction
 * with the LMetric class.  Be sure to use the same template parameters for
 * LMetric as you do for HRectBound -- otherwise odd results may occur.
 *
 * @tparam Power The metric to use; use 2 for Euclidean (L2).
 * @tparam TakeRoot Whether or not the root should be taken (see LMetric
 *     documentation).
 */
template<int Power = 2, bool TakeRoot = false>
class HRectBound
{
 public:
  /**
   * Empty constructor; creates a bound of dimensionality 0.
   */
  HRectBound();

  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   */
  HRectBound(const size_t dimension);

  /***
   * Copy constructor; necessary to prevent memory leaks.
   */
  HRectBound(const HRectBound& other);
  HRectBound& operator=(const HRectBound& other); // Same as copy constructor.

  /**
   * Destructor: clean up memory.
   */
  ~HRectBound();

  /**
   * Resets all dimensions to the empty set (so that this bound contains
   * nothing).
   */
  void Clear();

  /** Gets the dimensionality */
  size_t Dim() const { return dim; }

  /**
   * Sets and gets the range for a particular dimension.
   */
  math::Range& operator[](const size_t i);
  const math::Range& operator[](const size_t i) const;

  /**
   * Calculates the centroid of the range, placing it into the given vector.
   *
   * @param centroid Vector which the centroid will be written to.
   */
  void Centroid(arma::vec& centroid) const;

  /**
   * Calculates minimum bound-to-point distance.
   *
   * @param point Point to which the minimum distance is requested.
   */
  template<typename VecType>
  double MinDistance(const VecType& point) const;

  /**
   * Calculates minimum bound-to-bound distance.
   *
   * @param other Bound to which the minimum distance is requested.
   */
  double MinDistance(const HRectBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   *
   * @param point Point to which the maximum distance is requested.
   */
  template<typename VecType>
  double MaxDistance(const VecType& point) const;

  /**
   * Computes maximum distance.
   *
   * @param other Bound to which the maximum distance is requested.
   */
  double MaxDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * @param other Bound to which the minimum and maximum distances are
   *     requested.
   */
  math::Range RangeDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   *
   * @param point Point to which the minimum and maximum distances are
   *     requested.
   */
  template<typename VecType>
  math::Range RangeDistance(const VecType& point) const;

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
   */
  template<typename VecType>
  bool Contains(const VecType& point) const;

 private:
  //! The dimensionality of the bound.
  size_t dim;
  //! The bounds for each dimension.
  math::Range* bounds;
};

}; // namespace bound
}; // namespace mlpack

#include "hrectbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_HRECTBOUND_HPP
