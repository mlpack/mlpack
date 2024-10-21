/**
 * @file core/tree/binary_space_tree/rp_tree_mean_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (RPTreeMeanSplit) to split a binary space partition
 * tree.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_IMPL_HPP

#include "rp_tree_max_split.hpp"

namespace mlpack {

template<typename BoundType, typename MatType>
bool RPTreeMeanSplit<BoundType, MatType>::SplitNode(const BoundType& bound,
                                                    MatType& data,
                                                    const size_t begin,
                                                    const size_t count,
                                                    SplitInfo& splitInfo)
{
  const size_t maxNumSamples = 100;
  const size_t numSamples = std::min(maxNumSamples, count);

  // Get no more than numSamples distinct samples.
  arma::uvec samples;
  if (numSamples < count)
    samples = begin + arma::randperm(count, numSamples);
  else
    samples = begin + arma::linspace<arma::uvec>(0, count - 1, count);

  // Find the average distance between points.
  ElemType averageDistanceSq = GetAveragePointDistance(data, samples);

  const ElemType threshold = 10;

  if (bound.Diameter() * bound.Diameter() <= threshold * averageDistanceSq)
  {
    // We will perform the median split.
    splitInfo.meanSplit = false;

    splitInfo.direction.zeros(data.n_rows);

    // Get a random normal vector.
    RandVector(splitInfo.direction);

    // Get the median value of the scalar products of the normal and the
    // sampled points. The node will be split according to this value.
    return GetDotMedian(data, samples, splitInfo.direction, splitInfo.splitVal);
  }
  else
  {
    // We will perform the mean split.
    splitInfo.meanSplit = true;

    // Get the median of the distances between the mean point and the sampled
    // points. The node will be split according to this value.
    return GetMeanMedian(data, samples, splitInfo.mean, splitInfo.splitVal);
  }
}

template<typename BoundType, typename MatType>
typename MatType::elem_type RPTreeMeanSplit<BoundType, MatType>::
GetAveragePointDistance(
    MatType& data,
    const arma::uvec& samples)
{
  ElemType dist = 0;

  for (size_t i = 0; i < samples.n_elem; ++i)
    for (size_t j = i + 1; j < samples.n_elem; ++j)
      dist += SquaredEuclideanDistance::Evaluate(data.col(samples[i]),
          data.col(samples[j]));

  dist /= (samples.n_elem * (samples.n_elem - 1) / 2);

  return dist;
}

template<typename BoundType, typename MatType>
bool RPTreeMeanSplit<BoundType, MatType>::GetDotMedian(
    const MatType& data,
    const arma::uvec& samples,
    const arma::Col<ElemType>& direction,
    ElemType& splitVal)
{
  arma::Col<ElemType> values(samples.n_elem);

  for (size_t k = 0; k < samples.n_elem; ++k)
    values[k] = dot(data.col(samples[k]), direction);

  const ElemType maximum = arma::max(values);
  const ElemType minimum = min(values);
  if (minimum == maximum)
    return false;

  splitVal = arma::median(values);

  if (splitVal == maximum)
    splitVal = minimum;

  return true;
}

template<typename BoundType, typename MatType>
bool RPTreeMeanSplit<BoundType, MatType>::GetMeanMedian(
    const MatType& data,
    const arma::uvec& samples,
    arma::Col<ElemType>& mean,
    ElemType& splitVal)
{
  arma::Col<ElemType> values(samples.n_elem);

  mean = arma::mean(data.cols(samples), 1);

  arma::Col<ElemType> tmp(data.n_rows);

  for (size_t k = 0; k < samples.n_elem; ++k)
  {
    tmp = data.col(samples[k]);
    tmp -= mean;

    values[k] = dot(tmp, tmp);
  }

  const ElemType maximum = arma::max(values);
  const ElemType minimum = min(values);
  if (minimum == maximum)
    return false;

  splitVal = arma::median(values);

  if (splitVal == maximum)
    splitVal = minimum;

  return true;
}

} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_IMPL_HPP
