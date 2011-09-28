/**
 * @file tree/dhrectbound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the DHrectBound policy, which
 * implements a hyperrectangle bound.
 *
 * @experimental
 */

#ifndef TREE_DHRECTBOUND_H
#define TREE_DHRECTBOUND_H

#include <armadillo>

namespace mlpack {
namespace bound {

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class HRectBound {
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

  /**
   * Destructor: clean up memory.
   */
  ~HRectBound();

  /**
   * Resets all dimensions to the empty set (so that this bound contains
   * nothing).
   */
  void Clear();

  /**
   * Sets the dimensionality of the bound.
   */
  void SetSize(size_t dim);

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const arma::vec& point) const;

  /** Gets the dimensionality */
  size_t dim() const { return dim_; }

  /**
   * Sets and gets the range for a particular dimension.
   */
  Range& operator[](size_t i);
  const Range operator[](size_t i) const;

  /**
   * Calculates the centroid of the range, placing it into the given vector.
   *
   * @param centroid Vector which the centroid will be written to.
   */
  void Centroid(arma::vec& centroid) const;

  /**
   * Calculates minimum bound-to-bound squared distance, with
   * an offset between their respective coordinate systems.
   */
  double MinDistance(const HRectBound& other, const arma::vec& offset) const;

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistance(const arma::vec& point) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistance(const HRectBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistance(const arma::vec& point) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const HRectBound& other) const;

  /**
   * Computes maximum distance with offset
   */
  double MaxDistance(const HRectBound& other, const arma::vec& offset) const;

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  Range RangeDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
   */
  Range RangeDistance(const arma::vec& point) const;

  /**
   * Computes minimax distance, where the other node is trying to avoid me.
   */
  double MinimaxDistance(const HRectBound& other) const;

  /**
   * Calculates midpoint-to-midpoint bounding box distance.
   */
  double MidDistance(const HRectBound& other) const;

  /**
   * Expands this region to include a new point.
   */
  HRectBound& operator|=(const arma::vec& vector);

  /**
   * Expands this region to encompass another bound.
   */
  HRectBound& operator|=(const HRectBound& other);

 private:
  Range *bounds_;
  size_t dim_;
};

}; // namespace bound
}; // namespace mlpack

#include "dhrectbound_impl.h"

#endif
