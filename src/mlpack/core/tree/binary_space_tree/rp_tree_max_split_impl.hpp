/**
 * @file rp_tree_max_split_impl.hpp
 * @author Mikhail Lozhnikov
 *
 * Implementation of class (RPTreeMaxSplit) to split a binary space partition
 * tree.
 */
#ifndef MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP
#define MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP

#include "rp_tree_max_split.hpp"

namespace mlpack {
namespace tree {

template<typename BoundType, typename MatType>
bool RPTreeMaxSplit<BoundType, MatType>::SplitNode(const BoundType& /* bound */,
                                                  MatType& data,
                                                  const size_t begin,
                                                  const size_t count,
                                                  SplitInfo& splitInfo)
{
  splitInfo.direction.zeros(data.n_rows);

  // Get the normal to the hyperplane.
  GetRandomDirection(splitInfo.direction);

  // Get the value according to which we will perform the split.
  if (!GetSplitVal(data, begin, count, splitInfo.direction, splitInfo.splitVal))
    return false;

  return true;
}

template<typename BoundType, typename MatType>
void RPTreeMaxSplit<BoundType, MatType>::GetRandomDirection(
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
typename MatType::elem_type RPTreeMaxSplit<BoundType, MatType>::
GetRandomDeviation(const MatType& data,
                   const size_t begin,
                   const size_t count,
                   const arma::Col<ElemType>& direction)
{
  // Choose a random point
  size_t index = math::RandInt(begin, begin + count);

  ElemType furthestDistance = 0;

  // Find the furthest point from the point that we chose
  for (size_t i = begin; i < index; i++)
  {
    const ElemType dist = metric::SquaredEuclideanDistance::Evaluate(
        data.col(index), data.col(i));
    if (dist > furthestDistance)
      furthestDistance = dist;
  }

  for (size_t i = index; i < begin + count; i++)
  {
    const ElemType dist = metric::SquaredEuclideanDistance::Evaluate(
        data.col(index), data.col(i));
    if (dist > furthestDistance)
      furthestDistance = dist;
  }

  // Get a random deviation.
  return math::Random(-6.0 * std::sqrt(furthestDistance / data.n_rows),
                      6.0 * std::sqrt(furthestDistance / data.n_rows));
}

template<typename BoundType, typename MatType>
void RPTreeMaxSplit<BoundType, MatType>::GetDistinctSamples(
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
bool RPTreeMaxSplit<BoundType, MatType>::GetSplitVal(
    const MatType& data,
    const size_t begin,
    const size_t count,
    const arma::Col<ElemType>& direction,
    ElemType& splitVal)
{
  const size_t maxNumSamples = 100;
  const size_t numSamples = std::min(maxNumSamples, count);
  arma::uvec samples;

  // Get no more than numSamples distinct samples.
  GetDistinctSamples(samples, begin, count, numSamples);

  std::vector<ElemType> values(samples.n_elem);

  // Find the median of scalar products of the samples and the normal vector.
  for (size_t k = 0; k < samples.n_elem; k++)
    values[k] = arma::dot(data.col(samples[k]), direction);

  std::sort(values.begin(), values.end());

  if (values[0] == values[values.size() - 1])
    return false;

  splitVal = values[values.size() / 2];

  // Add a random deviation to the median.
  // This algorithm differs from the method suggested in the
  // random projection tree paper.
  splitVal += math::Random((values[values.size() / 2] - values[0]) * 0.75,
      (values[values.size() - 1] - values[values.size() / 2]) * 0.75);

  return true;
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP
