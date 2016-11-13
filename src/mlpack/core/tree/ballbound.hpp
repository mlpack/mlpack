/**
 * @file ballbound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a ball bound that works in arbitrary metric spaces.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BALLBOUND_HPP
#define MLPACK_CORE_TREE_BALLBOUND_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include "bound_traits.hpp"

namespace mlpack {
namespace bound {

/**
 * Ball bound encloses a set of points at a specific distance (radius) from a
 * specific point (center). MetricType is the custom metric type that defaults
 * to the Euclidean (L2) distance.
 *
 * @tparam MetricType metric type used in the distance measure.
 * @tparam VecType Type of vector (arma::vec or arma::sp_vec or similar).
 */
template<typename MetricType = metric::LMetric<2, true>,
         typename VecType = arma::vec>
class BallBound
{
 public:
  //! The underlying data type.
  typedef typename VecType::elem_type ElemType;
  //! A public version of the vector type.
  typedef VecType Vec;

 private:
  //! The radius of the ball bound.
  ElemType radius;
  //! The center of the ball bound.
  VecType center;
  //! The metric used in this bound.
  MetricType* metric;

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
  BallBound(const ElemType radius, const VecType& center);

  //! Copy constructor. To prevent memory leaks.
  BallBound(const BallBound& other);

  //! For the same reason as the copy constructor: to prevent memory leaks.
  BallBound& operator=(const BallBound& other);

  //! Move constructor: take possession of another bound.
  BallBound(BallBound&& other);

  //! Destructor to release allocated memory.
  ~BallBound();

  //! Get the radius of the ball.
  ElemType Radius() const { return radius; }
  //! Modify the radius of the ball.
  ElemType& Radius() { return radius; }

  //! Get the center point of the ball.
  const VecType& Center() const { return center; }
  //! Modify the center point of the ball.
  VecType& Center() { return center; }

  //! Get the dimensionality of the ball.
  size_t Dim() const { return center.n_elem; }

  /**
   * Get the minimum width of the bound (this is same as the diameter).
   * For ball bounds, width along all dimensions remain same.
   */
  ElemType MinWidth() const { return radius * 2.0; }

  //! Get the range in a certain dimension.
  math::RangeType<ElemType> operator[](const size_t i) const;

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const VecType& point) const;

  /**
   * Place the center of BallBound into the given vector.
   *
   * @param center Vector which the centroid will be written to.
   */
  void Center(VecType& center) const { center = this->center; }

  /**
   * Calculates minimum bound-to-point squared distance.
   */
  template<typename OtherVecType>
  ElemType MinDistance(const OtherVecType& point,
                       typename boost::enable_if<IsVector<OtherVecType>>* = 0)
      const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   */
  ElemType MinDistance(const BallBound& other) const;

  /**
   * Computes maximum distance.
   */
  template<typename OtherVecType>
  ElemType MaxDistance(const OtherVecType& point,
                       typename boost::enable_if<IsVector<OtherVecType>>* = 0)
      const;

  /**
   * Computes maximum distance.
   */
  ElemType MaxDistance(const BallBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   */
  template<typename OtherVecType>
  math::RangeType<ElemType> RangeDistance(
      const OtherVecType& other,
      typename boost::enable_if<IsVector<OtherVecType>>* = 0) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum distance.
   */
  math::RangeType<ElemType> RangeDistance(const BallBound& other) const;

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
  ElemType Diameter() const { return 2 * radius; }

  //! Returns the distance metric used in this bound.
  const MetricType& Metric() const { return *metric; }
  //! Modify the distance metric used in this bound.
  MetricType& Metric() { return *metric; }

  //! Serialize the bound.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);
};

//! A specialization of BoundTraits for this bound type.
template<typename MetricType, typename VecType>
struct BoundTraits<BallBound<MetricType, VecType>>
{
  //! These bounds are potentially loose in some dimensions.
  const static bool HasTightBounds = false;
};

} // namespace bound
} // namespace mlpack

#include "ballbound_impl.hpp"

#endif // MLPACK_CORE_TREE_DBALLBOUND_HPP
