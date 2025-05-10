/**
 * @file core/tree/hollow_ball_bound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a hollow ball bound that works in arbitrary metric spaces.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
#define MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include "bound_traits.hpp"

namespace mlpack {

/**
 * Hollow ball bound encloses a set of points at a specific distance (radius)
 * from a specific point (center) except points at a specific distance from
 * another point (the center of the hole). DistanceType is the custom distance
 * metric type that defaults to the Euclidean (L2) distance.
 *
 * @tparam TDistanceType metric type used in the distance measure.
 * @tparam ElemType Type of element (float or double or similar).
 */
template<typename TDistanceType = LMetric<2, true>,
         typename ElemType = double>
class HollowBallBound
{
 public:
  //! A public version of the metric type.
  using DistanceType = TDistanceType;

 private:
  //! The inner and the outer radii of the bound.
  RangeType<ElemType> radii;
  //! The center of the ball bound.
  arma::Col<ElemType> center;
  //! The center of the hollow.
  arma::Col<ElemType> hollowCenter;
  //! The distance metric used in this bound.
  DistanceType* distance;

  /**
   * To know whether this object allocated memory to the distance member
   * variable. This will be true except in the copy constructor and the
   * overloaded assignment operator. We need this to know whether we should
   * delete the distance member variable in the destructor.
   */
  bool ownsDistance;

 public:
  //! Empty Constructor.
  HollowBallBound();

  /**
   * Create the ball bound with the specified dimensionality.
   *
   * @param dimension Dimensionality of ball bound.
   */
  HollowBallBound(const size_t dimension);

  /**
   * Create the ball bound with the specified radius and center.
   *
   * @param innerRadius Inner radius of ball bound.
   * @param outerRadius Outer radius of ball bound.
   * @param center Center of ball bound.
   */
  template<typename VecType>
  HollowBallBound(const ElemType innerRadius,
                  const ElemType outerRadius,
                  const VecType& center);

  //! Copy constructor. To prevent memory leaks.
  HollowBallBound(const HollowBallBound& other);

  //! For the same reason as the copy constructor: to prevent memory leaks.
  HollowBallBound& operator=(const HollowBallBound& other);

  //! Move constructor: take possession of another bound.
  HollowBallBound(HollowBallBound&& other);

  //! Move assignment operator.
  HollowBallBound& operator=(HollowBallBound&& other);

  //! Destructor to release allocated memory.
  ~HollowBallBound();

  //! Get the outer radius of the ball.
  ElemType OuterRadius() const { return radii.Hi(); }
  //! Modify the outer radius of the ball.
  ElemType& OuterRadius() { return radii.Hi(); }

  //! Get the innner radius of the ball.
  ElemType InnerRadius() const { return radii.Lo(); }
  //! Modify the inner radius of the ball.
  ElemType& InnerRadius() { return radii.Lo(); }

  //! Get the center point of the ball.
  const arma::Col<ElemType>& Center() const { return center; }
  //! Modify the center point of the ball.
  arma::Col<ElemType>& Center() { return center; }

  //! Get the center point of the hollow.
  const arma::Col<ElemType>& HollowCenter() const { return hollowCenter; }
  //! Modify the center point of the hollow.
  arma::Col<ElemType>& HollowCenter() { return hollowCenter; }

  //! Get the dimensionality of the ball.
  size_t Dim() const { return center.n_elem; }

  /**
   * Get the minimum width of the bound (this is same as the diameter).
   * For ball bounds, width along all dimensions remain same.
   */
  ElemType MinWidth() const { return radii.Hi() * 2.0; }

  //! Get the range in a certain dimension.
  RangeType<ElemType> operator[](const size_t i) const;

  /**
   * Determines if a point is within this bound.
   *
   * @param point Point to check the condition.
   */
  template<typename VecType>
  bool Contains(const VecType& point) const;

  /**
   * Determines if another bound is within this bound.
   *
   * @param other Bound to check the condition.
   */
  bool Contains(const HollowBallBound& other) const;

  /**
   * Place the center of BallBound into the given vector.
   *
   * @param center Vector which the centroid will be written to.
   */
  template<typename VecType>
  void Center(VecType& center) const { center = this->center; }

  /**
   * Calculates minimum bound-to-point squared distance.
   *
   * @param point Point to which the minimum distance is requested.
   */
  template<typename VecType>
  ElemType MinDistance(const VecType& point,
                       typename std::enable_if_t<IsVector<VecType>::value>* = 0)
      const;

  /**
   * Calculates minimum bound-to-bound squared distance.
   *
   * @param other Bound to which the minimum distance is requested.
   */
  ElemType MinDistance(const HollowBallBound& other) const;

  /**
   * Computes maximum distance.
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
  ElemType MaxDistance(const HollowBallBound& other) const;

  /**
   * Calculates minimum and maximum bound-to-point distance.
   *
   * @param other Point to which the minimum and maximum distances are
   *     requested.
   */
  template<typename VecType>
  RangeType<ElemType> RangeDistance(
      const VecType& other,
      typename std::enable_if_t<IsVector<VecType>::value>* = 0) const;

  /**
   * Calculates minimum and maximum bound-to-bound distance.
   *
   * Example: bound1.MinDistanceSq(other) for minimum distance.
   *
   * @param other Bound to which the minimum and maximum distances are
   *     requested.
   */
  RangeType<ElemType> RangeDistance(const HollowBallBound& other) const;

  /**
   * Expand the bound to include the given point.  The centroid will not be
   * moved.
   *
   * @tparam MatType Type of matrix; could be arma::mat, arma::spmat, or a
   *     vector.
   * @tparam data Data points to add.
   */
  template<typename MatType>
  const HollowBallBound& operator|=(const MatType& data);

  /**
   * Expand the bound to include the given bound.  The centroid will not be
   * moved.
   *
   * @tparam MatType Type of matrix; could be arma::mat, arma::spmat, or a
   *     vector.
   * @tparam data Data points to add.
   */
  const HollowBallBound& operator|=(const HollowBallBound& other);

  /**
   * Returns the diameter of the ballbound.
   */
  ElemType Diameter() const { return 2 * radii.Hi(); }

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
template<typename DistanceType, typename ElemType>
struct BoundTraits<HollowBallBound<DistanceType, ElemType>>
{
  //! These bounds are potentially loose in some dimensions.
  static const bool HasTightBounds = false;
};

} // namespace mlpack

#include "hollow_ball_bound_impl.hpp"

#endif // MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
