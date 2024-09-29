/**
 * @file core/tree/binary_space_tree/rp_tree_max_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (RPTreeMaxSplit) to split a binary space partition
 * tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP

#include "rp_tree_max_split.hpp"
#include "rp_tree_mean_split.hpp"

namespace mlpack {

template<typename BoundType, typename MatType>
bool RPTreeMaxSplit<BoundType, MatType>::SplitNode(const BoundType& /* bound */,
                                                   MatType& data,
                                                   const size_t begin,
                                                   const size_t count,
                                                   SplitInfo& splitInfo)
{
  splitInfo.direction.zeros(data.n_rows);

  // Get the normal to the hyperplane.
  RandVector(splitInfo.direction);

  // Get the value according to which we will perform the split.
  return GetSplitVal(data, begin, count, splitInfo.direction,
      splitInfo.splitVal);
}

template<typename BoundType, typename MatType>
bool RPTreeMaxSplit<BoundType, MatType>::GetSplitVal(
    const MatType& data,
    const size_t begin,
    const size_t count,
    const arma::Col<ElemType>& direction,
    ElemType& splitVal)
{
  const size_t maxNumSamples = 100;
  const size_t numSamples = std::min(maxNumSamples, count);
  // Get no more than numSamples distinct samples.
  arma::uvec samples;
  if (numSamples < count)
    samples = begin + arma::randperm(count, numSamples);
  else
    samples = begin + arma::linspace<arma::uvec>(0, count - 1, count);

  arma::Col<ElemType> values(samples.n_elem);

  // Find the median of scalar products of the samples and the normal vector.
  for (size_t k = 0; k < samples.n_elem; ++k)
    values[k] = dot(data.col(samples[k]), direction);

  const ElemType maximum = arma::max(values);
  const ElemType minimum = min(values);
  if (minimum == maximum)
    return false;

  splitVal = arma::median(values);

  // Add a random deviation to the median.
  // This algorithm differs from the method suggested in the random projection
  // tree paper, for two reasons:
  //   1. Evaluating the method proposed in the paper is time-consuming, since
  //      we must solve the furthest-pair problem.
  //   2. The proposed method does not appear to guarantee that a valid split
  //      value will be generated (i.e. it can produce a split value where there
  //      may be no points on the left or the right).
  splitVal += Random((minimum - splitVal) * 0.75,
      (maximum - splitVal) * 0.75);

  if (splitVal == maximum)
    splitVal = minimum;

  return true;
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP
