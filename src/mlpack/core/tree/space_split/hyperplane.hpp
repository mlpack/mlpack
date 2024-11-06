/**
 * @file core/tree/space_split/hyperplane.hpp
 * @author Marcos Pividori
 *
 * Definition of Hyperplane and AxisOrthogonalHyperplane.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_HYPERPLANE_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_HYPERPLANE_HPP

#include <mlpack/prereqs.hpp>
#include "projection_vector.hpp"

namespace mlpack {

/**
 * HyperplaneBase defines a splitting hyperplane based on a projection vector
 * and projection value.
 *
 * @tparam BoundT The bound type considered.
 * @tparam ProjVectorT Type of projection vector (AxisParallelProjVector,
 *     ProjVector).
 */
template<typename BoundT, typename ProjVectorT>
class HyperplaneBase
{
 public:
  //! Useful typedef for the bound type.
  using BoundType = BoundT;
  //! Useful typedef for the projection vector type.
  using ProjVectorType = ProjVectorT;

 private:
  //! Projection vector.
  ProjVectorType projVect;

  //! Projection value that determines the decision boundary.
  double splitVal;

 public:
  /**
   * Empty Constructor. By default will consider all points to the left.
   */
  HyperplaneBase() :
      splitVal(DBL_MAX)
  {};

  /**
   * Create the hyperplane with the specified projection vector and split value.
   *
   * @param projVect Projection vector.
   * @param splitVal Split value.
   */
  HyperplaneBase(const ProjVectorType& projVect, double splitVal) :
      projVect(projVect),
      splitVal(splitVal)
  {};

  /**
   * Project the given point on the projection vector and subtract the
   * split value.
   *
   * @param point Point to be projected.
   */
  template<typename VecType>
  double Project(const VecType& point,
                 typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
  {
    if (splitVal == DBL_MAX)
      return 0;
    return projVect.Project(point) - splitVal;
  }

  /**
   * Determine if the given point is to the left of the hyperplane, this means
   * if the projection over the projection vector is negative or zero.
   *
   * @param point Point to be analyzed.
   */
  template<typename VecType>
  bool Left(const VecType& point,
            typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
  {
    return Project(point) <= 0;
  }

  /**
   * Determine if the given point is to the right of the hyperplane, this means
   * if the projection over the projection vector is positive.
   *
   * @param point Point to be analyzed.
   */
  template<typename VecType>
  bool Right(const VecType& point,
            typename std::enable_if_t<IsVector<VecType>::value>* = 0) const
  {
    return Project(point) > 0;
  }

  /**
   * Determine if the given bound is to the left of the hyperplane.
   *
   * @param bound Bound to be analyzed.
   */
  bool Left(const BoundType& bound) const
  {
    if (splitVal == DBL_MAX)
      return true;
    return projVect.Project(bound).Hi() <= splitVal;
  };

  /**
   * Determine if the given bound is to the right of the hyperplane.
   *
   * @param bound Bound to be analyzed.
   */
  bool Right(const BoundType& bound) const
  {
    if (splitVal == DBL_MAX)
      return false;
    return projVect.Project(bound).Lo() > splitVal;
  };

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(projVect));
    ar(CEREAL_NVP(splitVal));
  }
};

/**
 * AxisOrthogonalHyperplane represents a hyperplane orthogonal to an axis.
 */
template<typename DistanceType>
using AxisOrthogonalHyperplane = HyperplaneBase<HRectBound<DistanceType>,
    AxisParallelProjVector>;

/**
 * Hyperplane represents a general hyperplane (not necessarily axis-orthogonal).
 */
template<typename DistanceType>
using Hyperplane = HyperplaneBase<BallBound<DistanceType>, ProjVector>;

} // namespace mlpack

#endif
