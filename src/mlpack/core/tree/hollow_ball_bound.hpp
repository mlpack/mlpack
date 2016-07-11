/**
 * @file hollow_ball_bound.hpp
 *
 * Bounds that are useful for binary space partitioning trees.
 * Interface to a ball bound that works in arbitrary metric spaces.
 */
#ifndef MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
#define MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP

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
class HollowBallBound
{
 public:
  //! The underlying data type.
  typedef typename VecType::elem_type ElemType;
  //! A public version of the vector type.
  typedef VecType Vec;

 private:
  //! The radius of the inner ball bound.
  ElemType innerRadius;
  //! The radius of the outer ball bound.
  ElemType outerRadius;
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
   * @param innerRradius Inner radius of ball bound.
   * @param outerRradius Outer radius of ball bound.
   * @param center Center of ball bound.
   */
  HollowBallBound(const ElemType innerRadius,
                  const ElemType outerRadius,
                  const VecType& center);

  //! Copy constructor. To prevent memory leaks.
  HollowBallBound(const HollowBallBound& other);

  //! For the same reason as the copy constructor: to prevent memory leaks.
  HollowBallBound& operator=(const HollowBallBound& other);

  //! Move constructor: take possession of another bound.
  HollowBallBound(HollowBallBound&& other);

  //! Destructor to release allocated memory.
  ~HollowBallBound();

  //! Get the outer radius of the ball.
  ElemType OuterRadius() const { return outerRadius; }
  //! Modify the outer radius of the ball.
  ElemType& OuterRadius() { return outerRadius; }

  //! Get the innner radius of the ball.
  ElemType InnerRadius() const { return innerRadius; }
  //! Modify the inner radius of the ball.
  ElemType& InnerRadius() { return innerRadius; }

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
  ElemType MinWidth() const { return outerRadius * 2.0; }

  //! Get the range in a certain dimension.
  math::RangeType<ElemType> operator[](const size_t i) const;

  /**
   * Determines if a point is within this bound.
   */
  bool Contains(const VecType& point) const;

  /**
   * Determines if another bound is within this bound.
   */
  bool Contains(const HollowBallBound& other) const;

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
  ElemType MinDistance(const HollowBallBound& other) const;

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
  ElemType MaxDistance(const HollowBallBound& other) const;

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
  math::RangeType<ElemType> RangeDistance(const HollowBallBound& other) const;

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
  ElemType Diameter() const { return 2 * outerRadius; }

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
struct BoundTraits<HollowBallBound<MetricType, VecType>>
{
  //! These bounds are potentially loose in some dimensions.
  const static bool HasTightBounds = false;
};

} // namespace bound
} // namespace mlpack

#include "hollow_ball_bound_impl.hpp"

#endif // MLPACK_CORE_TREE_HOLLOW_BALL_BOUND_HPP
