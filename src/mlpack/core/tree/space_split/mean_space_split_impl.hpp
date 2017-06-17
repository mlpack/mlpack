/**
 * @file mean_space_split_impl.hpp
 * @author Marcos Pividori
 *
 * Implementation of MeanSpaceSplit, to create a splitting hyperplane
 * considering the midpoint/mean of the values in a certain projection.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_SPILL_TREE_MEAN_SPACE_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_SPILL_TREE_MEAN_SPACE_SPLIT_IMPL_HPP

#include "mean_space_split.hpp"
#include "space_split.hpp"

namespace mlpack {
namespace tree {

template<typename MetricType, typename MatType>
template<typename HyperplaneType>
bool MeanSpaceSplit<MetricType, MatType>::SplitSpace(
    const typename HyperplaneType::BoundType& bound,
    const MatType& data,
    const arma::Col<size_t>& points,
    HyperplaneType& hyp)
{
  typename HyperplaneType::ProjVectorType projVector;
  double midValue;

  if (!SpaceSplit<MetricType, MatType>::GetProjVector(bound, data, points,
      projVector, midValue))
    return false;

  double splitVal = 0.0;
  for (size_t i = 0; i < points.n_elem; i++)
    splitVal += projVector.Project(data.col(points[i]));
  splitVal /= points.n_elem;

  hyp = HyperplaneType(projVector, splitVal);

  return true;
}

} // namespace tree
} // namespace mlpack

#endif
