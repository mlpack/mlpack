/**
 * @file ballbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a ball bound that works in arbitrary metric spaces.
 *
 * This file is part of MLPACK 1.0.9.
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

#ifndef __MLPACK_CORE_TREE_BALLBOUND_HPP
#define __MLPACK_CORE_TREE_BALLBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace bound {

/**
 * Ball bound encloses a set of points at a specific distance (radius) from a
 * specific point (center). TMetricType is the custom metric type that defaults
 * to the Euclidean (L2) distance.
 *
 * @tparam VecType Type of vector (arma::vec or arma::sp_vec).
 * @tparam TMetricType metric type used in the distance measure.
 */
template<typename VecType = arma::vec,
         typename TMetricType = metric::LMetric<2, true> >
class BallBound
{
 public:
  typedef VecType Vec;
  //! Need this for Binary Space Partion Tree
  typedef TMetricType MetricType;

 private:

  //! The radius of the ball bound.
  double radius;

  //! The center of the ball bound.
  VecType center;

  //! The metric used in this bound.
  TMetricType* metric;

  /**
   * To know whether this object allocated memory to the metric member
   * variable. This will be true except in the copy constructor and the
   * overloaded assignment operator. We need this to know whether we should
   * delete the metric member variable in the destructor.
   */
  bool ownsMetric;

 public:

  //! Empty Constructor.
  BallBound();

  /**
   * Create the ball bound with the specified dimensionality.
   *
   * @param dimension Dimensionality of ball bound.
   */
  BallBound(const size_t dimension);

  /**
   * Create the ball bound with the specified radius and center.
   *
   * @param radius Radius of ball bound.
   * @param center Center of ball bound.
   */
  BallBound(const double radius, const VecType& center);

  //! Copy constructor. To prevent memory leaks.
  BallBound(const BallBound& other);

  //! For the same reason as the Copy Constructor. To prevent memory leaks.
  BallBound& operator=(const BallBound& other);

  //! Destructor to release allocated memory.
  ~BallBound();

  //! Get the radius of the ball.
  double Radius() const { return radius; }
  //! Modify the radius of the ball.
  double& Radius() { return radius; }

  //! Get the center point of the ball.
  const VecType& Center() const { return center; }
  //! Modify the center point of the ball.
  VecType& Center() { return center; }

  //! Get the dimensionality of the ball.
  double Dim() const { return center.n_elem; }

  /**
   * Get the minimum width of the bound (this is same as the diameter).
   * For ball bounds, width along all dimensions remain same.
   */
  double MinWidth() const { return radius * 2.0; }

  //! Get the range in a certain dimension.
  math::Range operator[](const size_t i) const;

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const VecType& point) const;

  /**
   * Place the centroid of BallBound into the given vector.
   *
   * @param centroid Vector which the centroid will be written to.
   */
  void Centroid(VecType& centroid) const { centroid = center; }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  template<typename OtherVecType>
  double MinDistance(const OtherVecType& point,
                     typename boost::enable_if<IsVector<OtherVecType> >* = 0)
      const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   */
  double MinDistance(const BallBound& other) const;

  /**
   * Computes maximum distance.
   */
  template<typename OtherVecType>
  double MaxDistance(const OtherVecType& point,
                     typename boost::enable_if<IsVector<OtherVecType> >* = 0)
      const;

  /**
   * Computes maximum distance.
   */
  double MaxDistance(const BallBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   */
  template<typename OtherVecType>
  math::Range RangeDistance(
      const OtherVecType& other,
      typename boost::enable_if<IsVector<OtherVecType> >* = 0) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum distance.
   */
  math::Range RangeDistance(const BallBound& other) const;

  /**
   * Expand the bound to include the given node.
   */
  const BallBound& operator|=(const BallBound& other);

  /**
   * Expand the bound to include the given point.  The centroid is recalculated
   * to be the center of all of the given points.
   *
   * @tparam MatType Type of matrix; could be arma::mat, arma::spmat, or a
   *     vector.
   * @tparam data Data points to add.
   */
  template<typename MatType>
  const BallBound& operator|=(const MatType& data);

  /**
   * Returns the diameter of the ballbound.
   */
  double Diameter() const { return 2 * radius; }

  /**
   * Returns the distance metric used in this bound.
   */
  TMetricType Metric() const { return *metric; }

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;

};

}; // namespace bound
}; // namespace mlpack

#include "ballbound_impl.hpp"

#endif // __MLPACK_CORE_TREE_DBALLBOUND_HPP
