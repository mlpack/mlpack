/**
 * @file vantage_point_split_impl.hpp
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
namespace tree {

template<typename BoundType, typename MatType, size_t MaxNumSamples>
bool VantagePointSplit<BoundType, MatType, MaxNumSamples>::
SplitNode(const BoundType& bound, MatType& data, const size_t begin,
    const size_t count, SplitInfo& splitInfo)
{
  ElemType mu = 0;
  size_t vantagePointIndex;

  // Find the best vantage point.
  SelectVantagePoint(bound.Metric(), data, begin, count, vantagePointIndex, mu);

  // If all points are equal, we can't split.
  if (mu == 0)
    return false;

  splitInfo = SplitInfo(bound.Metric(), data.col(vantagePointIndex), mu);

  return true;
}

template<typename BoundType, typename MatType, size_t MaxNumSamples>
void VantagePointSplit<BoundType, MatType, MaxNumSamples>::
SelectVantagePoint(const MetricType& metric, const MatType& data,
    const size_t begin, const size_t count, size_t& vantagePoint, ElemType& mu)
{
  arma::uvec vantagePointCandidates;
  arma::Col<ElemType> distances(MaxNumSamples);

  // Get no more than max(MaxNumSamples, count) vantage point candidates
  math::ObtainDistinctSamples(begin, begin + count, MaxNumSamples,
      vantagePointCandidates);

  ElemType bestSpread = 0;

  arma::uvec samples;
  //  Evaluate each candidate
  for (size_t i = 0; i < vantagePointCandidates.n_elem; i++)
  {
    // Get no more than min(MaxNumSamples, count) random samples
    math::ObtainDistinctSamples(begin, begin + count, MaxNumSamples, samples);

    // Calculate the second moment of the distance to the vantage point
    // candidate using these random samples.
    distances.set_size(samples.n_elem);

    for (size_t j = 0; j < samples.n_elem; j++)
      distances[j] = metric.Evaluate(data.col(vantagePointCandidates[i]),
          data.col(samples[j]));

    const ElemType spread = arma::sum(distances % distances) / samples.n_elem;

    if (spread > bestSpread)
    {
      bestSpread = spread;
      vantagePoint = vantagePointCandidates[i];
      // Calculate the median value of the distance from the vantage point
      // candidate to these samples.
      mu = arma::median(distances);
    }
  }
  assert(bestSpread > 0);
}

} // namespace tree
} // namespace mlpack

#endif  // MLPACK_CORE_TREE_BINARY_SPACE_TREE_VANTAGE_POINT_SPLIT_IMPL_HPP
