/**
 * @file core/tree/space_split/mean_space_split.hpp
 * @author Marcos Pividori
 *
 * Definition of MeanSpaceSplit, to create a splitting hyperplane considering
 * the mean of the values in a certain projection.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_MEAN_SPACE_SPLIT_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_MEAN_SPACE_SPLIT_HPP

#include <mlpack/prereqs.hpp>
#include "hyperplane.hpp"

namespace mlpack {

template<typename DistanceType, typename MatType>
class MeanSpaceSplit
{
 public:
  /**
   * Create a splitting hyperplane considering the mean of the values in a
   * certain projection.
   *
   * @param bound The bound used for this node.
   * @param data The dataset used by the tree.
   * @param points Vector of indexes of points to be considered.
   * @param hyp Resulting splitting hyperplane.
   * @return Flag to determine if split is possible.
   */
  template<typename HyperplaneType>
  static bool SplitSpace(
      const typename HyperplaneType::BoundType& bound,
      const MatType& data,
      const arma::Col<size_t>& points,
      HyperplaneType& hyp);
};

} // namespace mlpack

// Include implementation.
#include "mean_space_split_impl.hpp"

#endif
