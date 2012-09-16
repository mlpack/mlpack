/**
 * @file hrectbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 *
 * This file describes the interface for the DHrectBound policy, which
 * implements a hyperrectangle bound.
 *
 * @experimental
 *
 * This file is part of MLPACK 1.0.3.
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
   * Calculates minimum bound-to-point squared distance.
   */
  template<typename VecType>
  double MinDistance(const VecType& point) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum squared distance.
   */
  double MinDistance(const HRectBound& other) const;

  /**
   * Calculates maximum bound-to-point squared distance.
   */
  template<typename VecType>
  double MaxDistance(const VecType& point) const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-bound squared distance.
   */
  math::Range RangeDistance(const HRectBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point squared distance.
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
  size_t dim;
  math::Range* bounds;
};

}; // namespace bound
}; // namespace mlpack

#include "hrectbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_HRECTBOUND_HPP
