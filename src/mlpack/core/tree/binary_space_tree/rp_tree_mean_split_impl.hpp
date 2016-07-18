/**
 * @file rp_tree_mean_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (RPTreeMeanSplit) to split a binary space partition
 * tree.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_IMPL_HPP

#include "rp_tree_max_split.hpp"

namespace mlpack {
namespace tree {

template<typename BoundType, typename MatType>
bool RPTreeMeanSplit<BoundType, MatType>::SplitNode(const BoundType&  bound,
                                                  MatType& data,
                                                  const size_t begin,
                                                  const size_t count,
                                                  SplitInfo& splitInfo)
{
  const size_t maxNumSamples = 100;
  const size_t numSamples = std::min(maxNumSamples, count);
  arma::uvec samples;

  // Get no more than numSamples distinct samples.
  GetDistinctSamples(samples, begin, count, numSamples);

  // Find the average distance between points.
  ElemType averageDistanceSq = GetAveragePointDistance(data, samples);

  const ElemType threshold = 10;

  if (bound.Diameter() * bound.Diameter() <= threshold * averageDistanceSq)
  {
    // We will perform the median split.
    splitInfo.meanSplit = false;

    splitInfo.direction.zeros(data.n_rows);

    // Get a random normal vector.
    GetRandomDirection(splitInfo.direction);

    // Get the median value of the scalar products of the normal and the
    // sampled points. The node will be split according to this value.
    if (!GetDotMedian(data, samples, splitInfo.direction, splitInfo.splitVal))
      return false;
  }
  else
  {
    // We will perform the mean split.
    splitInfo.meanSplit = true;

    // Get the median of the distances between the mean point and the sampled
    // points. The node will be split according to this value.
    if (!GetMeanMedian(data, samples, splitInfo.mean, splitInfo.splitVal))
      return false;
  }

  return true;
}

template<typename BoundType, typename MatType>
void RPTreeMeanSplit<BoundType, MatType>::GetDistinctSamples(
    arma::uvec& distinctSamples,
    const size_t begin,
    const size_t count,
    const size_t numSamples)
{
  arma::Col<size_t> samples;

  samples.zeros(count);

  for (size_t i = 0; i < numSamples; i++)
    samples [ (size_t) math::RandInt(count) ]++;

  distinctSamples = arma::find(samples > 0);

  distinctSamples += begin;
}

template<typename BoundType, typename MatType>
typename MatType::elem_type RPTreeMeanSplit<BoundType, MatType>::
GetAveragePointDistance(
    MatType& data,
    const arma::uvec& samples)
{
  ElemType dist = 0;

  for (size_t i = 0; i < samples.n_elem; i++)
    for (size_t j = i + 1; j < samples.n_elem; j++)
      dist += metric::SquaredEuclideanDistance::Evaluate(data.col(samples[i]),
          data.col(samples[j]));

  dist /= (samples.n_elem * (samples.n_elem - 1) / 2);

  return dist;
}

template<typename BoundType, typename MatType>
void RPTreeMeanSplit<BoundType, MatType>::GetRandomDirection(
    arma::Col<ElemType>& direction)
{
  arma::Col<ElemType> origin;

  origin.zeros(direction.n_rows);

  for (size_t k = 0; k < direction.n_rows; k++)
    direction[k] = math::Random(-1.0, 1.0);

  ElemType length = metric::EuclideanDistance::Evaluate(origin, direction);

  if (length > 0)
    direction /= length;
  else
  {
    // If the vector is equal to 0, choose an arbitrary dimension.
    size_t k = math::RandInt(direction.n_rows);

    direction[k] = 1.0;

    length = metric::EuclideanDistance::Evaluate(origin, direction);

    direction[k] /= length;
  }
}

template<typename BoundType, typename MatType>
bool RPTreeMeanSplit<BoundType, MatType>::GetDotMedian(
    const MatType& data,
    const arma::uvec& samples,
    const arma::Col<ElemType>& direction,
    ElemType& splitVal)
{
  std::vector<ElemType> values(samples.n_elem);

  for (size_t k = 0; k < samples.n_elem; k++)
    values[k] = arma::dot(data.col(samples[k]), direction);

  std::sort(values.begin(), values.end());

  if (values[0] == values[values.size() - 1])
    return false;

  splitVal = values[values.size() / 2];

  return true;
}

template<typename BoundType, typename MatType>
bool RPTreeMeanSplit<BoundType, MatType>::GetMeanMedian(
    const MatType& data,
    const arma::uvec& samples,
    arma::Col<ElemType>& mean,
    ElemType& splitVal)
{
  std::vector<ElemType> values(samples.n_elem);

  mean.zeros(data.n_rows);

  for (size_t k = 0; k < samples.n_elem; k++)
    mean += data.col(samples[k]);

  mean /= samples.n_elem;
  arma::Col<ElemType> tmp(data.n_elem);

  for (size_t k = 0; k < samples.n_elem; k++)
  {
    tmp = data.col(samples[k]);
    tmp -= mean;

    values[k] = arma::dot(tmp, tmp);
  }

  std::sort(values.begin(), values.end());

  if (values[0] == values[values.size() - 1])
    return false;

  splitVal = values[values.size() / 2];

  return true;
}



} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MEAN_SPLIT_IMPL_HPP
