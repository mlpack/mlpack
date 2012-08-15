/**
 * @file tree/periodichrectbound.h
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the PeriodicHRectBound policy, which
 * implements a hyperrectangle bound in a periodic space.
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_TREE_PERIODICHRECTBOUND_HPP
#define __MLPACK_CORE_TREE_PERIODICHRECTBOUND_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bound {

/**
 * Hyper-rectangle bound for an L-metric.
 *
 * Template parameter t_pow is the metric to use; use 2 for Euclidean (L2).
 */
template<int t_pow = 2>
class PeriodicHRectBound
{
 public:
  /**
   * Empty constructor.
   */
  PeriodicHRectBound();

  /**
   * Specifies the box size.  The dimensionality is set to the same of the box
   * size, and the bounds are initialized to be empty.
   */
  PeriodicHRectBound(arma::vec box);

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
   * Modifies the box to the desired dimenstions.
   */
  void SetBoxSize(arma::vec box);

  /**
   * Returns the box vector.
   */
  const arma::vec& Box() const { return box; }

  /**
   * Resets all dimensions to the empty set.
   */
  void Clear();

  /** Gets the dimensionality */
  size_t Dim() const { return dim; }

  /**
   * Sets and gets the range for a particular dimension.
   */
  math::Range& operator[](size_t i);
  const math::Range operator[](size_t i) const;

  /***
   * Calculates the centroid of the range.  This does not factor in periodic
   * coordinates, so the centroid may not necessarily be inside the given box.
   *
   * @param centroid Vector to write the centroid to.
   */
  void Centroid(arma::vec& centroid) const;

  /**
   * Calculates minimum bound-to-point squared distance in the periodic bound
   * case.
   */
  double MinDistance(const arma::vec& point) const;

  /**
   * Calculates minimum bound-to-bound squared distance in the periodic bound
   * case.
   *
   * Example: bound1.MinDistance(other) for minimum squared distance.
   */
  double MinDistance(const PeriodicHRectBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance in the periodic bound
   * case.
   */
  double MaxDistance(const arma::vec& point) const;

  /**
   * Computes maximum bound-to-bound squared distance in the periodic bound
   * case.
   */
  double MaxDistance(const PeriodicHRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point squared distance in the
   * periodic bound case.
   */
  math::Range RangeDistance(const arma::vec& point) const;

  /**
   * Calculates minimum and maximum bound-to-bound squared distance in the
   * periodic bound case.
   */
  math::Range RangeDistance(const PeriodicHRectBound& other) const;

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
  math::Range *bounds;
  size_t dim;
  arma::vec box;
};

}; // namespace bound
}; // namespace mlpack

#include "periodichrectbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_PERIODICHRECTBOUND_HPP
