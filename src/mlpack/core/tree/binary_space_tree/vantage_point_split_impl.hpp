/**
 * @file core/tree/binary_space_tree/vantage_point_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (VantagePointSplit) to split a vantage point
 * tree according to the median value of the distance to a certain vantage point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP

#include "vantage_point_split.hpp"
#include <mlpack/core/tree/bounds.hpp>

namespace mlpack {

template<typename BoundType, typename MatType, size_t MaxNumSamples>
bool VantagePointSplit<BoundType, MatType, MaxNumSamples>::
SplitNode(const BoundType& bound, MatType& data, const size_t begin,
    const size_t count, SplitInfo& splitInfo)
{
  ElemType mu = 0;
  size_t vantagePointIndex = 0;

  // Find the best vantage point.
  SelectVantagePoint(bound.Distance(), data, begin, count, vantagePointIndex,
      mu);

  // If all points are equal, we can't split.
  if (mu == 0)
    return false;

  splitInfo = SplitInfo(bound.Distance(), data.col(vantagePointIndex), mu);

  return true;
}

template<typename BoundType, typename MatType, size_t MaxNumSamples>
void VantagePointSplit<BoundType, MatType, MaxNumSamples>::
SelectVantagePoint(const DistanceType& distance, const MatType& data,
    const size_t begin, const size_t count, size_t& vantagePoint, ElemType& mu)
{
  arma::Col<ElemType> distances(MaxNumSamples);

  // Get no more than max(MaxNumSamples, count) vantage point candidates
  arma::uvec vantagePointCandidates;
  if (MaxNumSamples > count)
  {
    vantagePointCandidates = begin + arma::linspace<arma::uvec>(0, count - 1,
        count);
  }
  else
  {
    vantagePointCandidates = begin + arma::randperm(count, MaxNumSamples);
  }

  ElemType bestSpread = 0;

  arma::uvec samples;
  //  Evaluate each candidate
  for (size_t i = 0; i < vantagePointCandidates.n_elem; ++i)
  {
    // Get no more than min(MaxNumSamples, count) random samples
    if (MaxNumSamples > count)
      samples = begin + arma::linspace<arma::uvec>(0, count - 1, count);
    else
      samples = begin + arma::randperm(count, MaxNumSamples);

    // Calculate the second moment of the distance to the vantage point
    // candidate using these random samples.
    distances.set_size(samples.n_elem);

    for (size_t j = 0; j < samples.n_elem; ++j)
      distances[j] = distance.Evaluate(data.col(vantagePointCandidates[i]),
          data.col(samples[j]));

    const ElemType spread = sum(distances % distances) / samples.n_elem;

    if (spread > bestSpread)
    {
      bestSpread = spread;
      vantagePoint = vantagePointCandidates[i];
      // Calculate the median value of the distance from the vantage point
      // candidate to these samples.
      mu = arma::median(distances);
    }
  }
}

} // namespace mlpack

#endif  // MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
