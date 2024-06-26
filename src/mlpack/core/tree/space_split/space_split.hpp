/**
 * @file core/tree/space_split/space_split.hpp
 * @author Marcos Pividori
 *
 * Definition of SpaceSplit, implementing some methods to create a projection
 * vector based on a given set of points.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_SPACE_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include "hyperplane.hpp"

namespace mlpack {

template<typename DistanceType, typename MatType>
class SpaceSplit
{
 public:
  /**
   * Create a projection vector based on the given set of point. This special
   * case will create an axis-parallel projection vector in the dimension that
   * has the maximum width.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the tree.
   * @param points Vector of indexes of points to be considered.
   * @param projVector Resulting axis-parallel projection vector.
   * @param midValue Mid value in the chosen projection.
   * @return Flag to determine if it is possible.
   */
  static bool GetProjVector(
      const HRectBound<DistanceType>& bound,
      const MatType& data,
      const arma::Col<size_t>& points,
      AxisParallelProjVector& projVector,
      double& midValue);

  /**
   * Create a projection vector based on the given set of point. We efficiently
   * estimate the farthest pair of points in the given set: p and q, and then
   * consider the projection vector (q - p).
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the tree.
   * @param points Vector of indexes of points to be considered.
   * @param projVector Resulting projection vector.
   * @param midValue Mid value in the chosen projection.
   * @return Flag to determine if it is possible.
   */
  template<typename BoundType>
  static bool GetProjVector(
      const BoundType& bound,
      const MatType& data,
      const arma::Col<size_t>& points,
      ProjVector& projVector,
      double& midValue);
};

} // namespace mlpack

// Include implementation.
#include "space_split_impl.hpp"

#endif
