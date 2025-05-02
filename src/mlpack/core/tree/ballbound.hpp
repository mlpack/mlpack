/**
 * @file core/tree/ballbound.hpp
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

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include "bound_traits.hpp"

namespace mlpack {

/**
 * Ball bound encloses a set of points at a specific distance (radius) from a
 * specific point (center). DistanceType is the custom distance metric type that
 * defaults to the Euclidean (L2) distance.
 *
 * @tparam DistanceType distance metric type used in the distance measure.
 * @tparam VecType Type of vector (arma::vec or arma::sp_vec or similar).
 */
template<typename DistanceType = LMetric<2, true>,
         typename ElemType = double,
         typename VecType = arma::Col<ElemType>>
class BallBound
{
 public:
  //! A public version of the vector type.
  using Vec = VecType;

 private:
  //! The radius of the ball bound.
  ElemType radius;
  //! The center of the ball bound.
  VecType center;
  //! The metric used in this bound.
  DistanceType* distance;

  /**
   * To know whether this object allocated memory to the distance metric member
   * variable. This will be true except in the copy constructor and the
   * overloaded assignment operator. We need this to know whether we should
   * delete the distance metric member variable in the destructor.
   */
  bool ownsDistance;

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

  //! Move assignment operator.
  BallBound& operator=(BallBound&& other);

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
  RangeType<ElemType> operator[](const size_t i) const;

  /**
   * Determines if a point is within this bound.
   *
   * @param point Point to check the condition.
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
   *
   * @param point Point to which the minimum distance is requested.
   */
  template<typename OtherVecType>
  ElemType MinDistance(
      const OtherVecType& point,
      typename std::enable_if_t<IsVector<OtherVecType>::value>* = 0) const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * @param other Bound to which the minimum distance is requested.
   */
  ElemType MinDistance(const BallBound& other) const;

  /**
   * Computes maximum distance.
   *
   * @param point Point to which the maximum distance is requested.
   */
  template<typename OtherVecType>
  ElemType MaxDistance(
      const OtherVecType& point,
      typename std::enable_if_t<IsVector<OtherVecType>::value>* = 0) const;

  /**
   * Computes maximum distance.
   *
   * @param other Bound to which the maximum distance is requested.
   */
  ElemType MaxDistance(const BallBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   *
   * @param other Point to which the minimum and maximum distances are
   *     requested.
   */
  template<typename OtherVecType>
  RangeType<ElemType> RangeDistance(
      const OtherVecType& other,
      typename std::enable_if_t<IsVector<OtherVecType>::value>* = 0) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum distance.
   *
   * @param other Bound to which the minimum and maximum distances are
   *     requested.
   */
  RangeType<ElemType> RangeDistance(const BallBound& other) const;

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
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  const DistanceType& Metric() const { return *distance; }
  //! Modify the distance metric used in this bound.
  [[deprecated("Will be removed in mlpack 5.0.0; use Distance()")]]
  DistanceType& Metric() { return *distance; }

  //! Returns the distance metric used in this bound.
  const DistanceType& Distance() const { return *distance; }
  //! Modify the distance metric used in this bound.
  DistanceType& Distance() { return *distance; }

  //! Serialize the bound.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t version);
};

//! A specialization of BoundTraits for this bound type.
template<typename DistanceType, typename VecType>
struct BoundTraits<BallBound<DistanceType, VecType>>
{
  //! These bounds are potentially loose in some dimensions.
  static const bool HasTightBounds = false;
};

} // namespace mlpack

#include "ballbound_impl.hpp"

#endif // MLPACK_CORE_TREE_DBALLBOUND_HPP
