/**
 * @file hrectbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the DHrectBound policy, which
 * implements a hyperrectangle bound.
 *
 * @experimental
 */
#ifndef __MLPACK_CORE_TREE_HRECTBOUND_HPP
#define __MLPACK_CORE_TREE_HRECTBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/math/range.hpp>

namespace mlpack {
namespace bound {

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
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
  HRectBound(size_t dimension);

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
  size_t dim() const { return dim_; }

  /**
   * Sets and gets the range for a particular dimension.
   */
  math::Range& operator[](size_t i);
  const math::Range operator[](size_t i) const;

  /**
   * Calculates the centroid of the range, placing it into the given vector.
   *
   * @param centroid Vector which the centroid will be written to.
   */
  void Centroid(arma::vec& centroid) const;

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const arma::vec& point) const;
  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const arma::vec& point,
                     const std::vector<size_t>& indices) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistance(const HRectBound& other) const;
  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistance(const HRectBound& other,
                     const std::vector<size_t>& indices) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistance(const arma::vec& point) const;
  /**
   * Computes maximum distance.
   */
  double MaxDistance(const arma::vec& point,
                     const std::vector<size_t>& indices) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const HRectBound& other) const;
  /**
   * Computes maximum distance.
   */
  double MaxDistance(const HRectBound& other,
                     const std::vector<size_t>& indices) const;

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  math::Range RangeDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
   */
  math::Range RangeDistance(const arma::vec& point) const;

  /**
   * Expands this region to include a new point.
   */
  HRectBound& operator|=(const arma::vec& vector);

  /**
   * Expands this region to encompass another bound.
   */
  HRectBound& operator|=(const HRectBound& other);

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const arma::vec& point) const;

 private:
  size_t dim_;
  math::Range *bounds_;
};

}; // namespace bound
}; // namespace mlpack

#include "hrectbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_HRECTBOUND_HPP
