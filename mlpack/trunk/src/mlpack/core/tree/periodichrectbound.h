/**
 * @file tree/periodichrectbound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the PeriodicHRectBound policy, which
 * implements a hyperrectangle bound in a periodic space.
 */
#ifndef __MLPACK_CORE_TREE_PERIODICHRECTBOUND_H
#define __MLPACK_CORE_TREE_PERIODICHRECTBOUND_H

#include <armadillo>

namespace mlpack {
namespace bound {

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class PeriodicHRectBound {
 public:
  /**
   * Empty constructor.
   */
  PeriodicHRectBound();

  /**
   * Specifies the box size, but not dimensionality.
   */
  PeriodicHRectBound(arma::vec box);

  /**
   * Initializes to specified dimensionality with each dimension the empty
   * set.
   */
  PeriodicHRectBound(size_t dimension, arma::vec box);

  /***
   * Copy constructor and copy operator.  These are necessary because we do our
   * own memory management.
   */
  PeriodicHRectBound(const PeriodicHRectBound& other);
  PeriodicHRectBound& operator=(const PeriodicHRectBound& other);

  /**
   * Destructor: clean up memory.
   */
  ~PeriodicHRectBound();

  /**
   * Modifies the box_size_ to the desired dimenstions.
   */
  void SetBoxSize(arma::vec box);

  /**
   * Returns the box_size_ vector.
   */
  arma::vec GetBoxSize();

  /**
   * Resets all dimensions to the empty set.
   */
  void Clear();

  /** Gets the dimensionality */
  size_t dim() const { return dim_; }

  /**
   * Sets and gets the range for a particular dimension.
   */
  Range& operator[](size_t i);
  const Range operator[](size_t i) const;

  /** Calculates the midpoint of the range */
  void Centroid(arma::vec& centroid) const;

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  double MinDistanceSq(const arma::vec& point) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistanceSq(const PeriodicHRectBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  double MaxDistanceSq(const arma::vec& point) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistanceSq(const PeriodicHRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
   */
  Range RangeDistanceSq(const arma::vec& point) const;

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  Range RangeDistanceSq(const PeriodicHRectBound& other) const;

  /**
   * Expands this region to include a new point.
   */
  PeriodicHRectBound& operator|=(const arma::vec& vector);

  /**
   * Expands this region to encompass another bound.
   */
  PeriodicHRectBound& operator|=(const PeriodicHRectBound& other);

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const arma::vec& point) const;

 private:
  Range *bounds_;
  size_t dim_;
  arma::vec box_size_;
};

}; // namespace bound
}; // namespace mlpack

#include "periodichrectbound_impl.h"

#endif
