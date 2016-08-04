/**
 * @file projection_vector.hpp
 * @author Marcos Pividori
 *
 * Definition of ProjVector and AxisParallelProjVector.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_PROJECTION_VECTOR_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_PROJECTION_VECTOR_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace tree {

/**
 * AxisParallelProjVector defines an axis-parallel projection vector.
 * We can efficiently project points, simply analyzing a specific dimension.
 */
class AxisParallelProjVector
{
  //! Dimension considered.
  size_t dim;

 public:
  /**
   * Create the projection vector based on the specified dimension.
   *
   * @param dim Dimension to be considered.
   */
  AxisParallelProjVector(size_t dim = 0) :
      dim(dim)
  {};

  /**
   * Project the given point on the projection vector.
   *
   * @param point Point to be projected.
   */
  template<typename VecType>
  double Project(const VecType& point,
                 typename boost::enable_if<IsVector<VecType> >::type* = 0) const
  {
    return point[dim];
  };

  /**
   * Project the given hrect bound on the projection vector.
   *
   * @param bound Bound to be projected.
   * @return Range of projected values.
   */
  template<typename MetricType>
  math::RangeType<double> Project(
      const bound::HRectBound<MetricType>& bound) const
  {
    return bound[dim];
  };

  /**
   * Project the given ball bound on the projection vector.
   *
   * @param bound Bound to be projected.
   * @return Range of projected values.
   */
  template<typename MetricType>
  math::RangeType<double> Project(
      const bound::BallBound<MetricType>& bound) const
  {
    return bound[dim];
  };

  /**
   * Serialization.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(dim, "dim");
  };
};

/**
 * ProjVector defines a general projection vector (not necessarily
 * axis-parallel).
 */
class ProjVector
{
  //! Projection vector.
  arma::vec projVect;

 public:
  /**
   * Empty Constructor.
   */
  ProjVector() :
      projVect()
  {};

  /**
   * Create the projection vector based on the specified vector.
   *
   * @param vect Vector to be considered.
   */
  ProjVector(const arma::vec& vect) :
      projVect(vect)
  {};

  /**
   * Project the given point on the projection vector.
   *
   * @param point Point to be projected.
   */
  template<typename VecType>
  double Project(const VecType& point,
                 typename boost::enable_if<IsVector<VecType> >::type* = 0) const
  {
    return arma::dot(point, projVect);
  };

  /**
   * Project the given ball bound on the projection vector.
   *
   * @param bound Bound to be projected.
   * @return Range of projected values.
   */
  template<typename MetricType>
  math::RangeType<double> Project(
      const bound::BallBound<MetricType>& bound) const
  {
    const double center = Project(bound.Center());
    const double radius = bound.Radius();
    return math::Range(center - radius, center + radius);
  };

  /**
   * Serialization.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(projVect, "projVect");
  };
};

} // namespace tree
} // namespace mlpack

#endif
