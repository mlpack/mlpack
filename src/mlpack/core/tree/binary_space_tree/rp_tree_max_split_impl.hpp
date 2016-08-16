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
#include "rp_tree_mean_split.hpp"

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
  RPTreeMeanSplit<BoundType, MatType>::GetRandomDirection(
      splitInfo.direction);

  // Get the value according to which we will perform the split.
  if (!GetSplitVal(data, begin, count, splitInfo.direction, splitInfo.splitVal))
    return false;

  return true;
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
  math::ObtainDistinctSamples(begin, begin + count, numSamples, samples);

  arma::Col<ElemType> values(samples.n_elem);

  // Find the median of scalar products of the samples and the normal vector.
  for (size_t k = 0; k < samples.n_elem; k++)
    values[k] = arma::dot(data.col(samples[k]), direction);

  const ElemType maximum = arma::max(values);
  const ElemType minimum = arma::min(values);
  if (minimum == maximum)
    return false;

  splitVal = arma::median(values);

  // Add a random deviation to the median.
  // This algorithm differs from the method suggested in the
  // random projection tree paper.
  splitVal += math::Random((minimum - splitVal) * 0.75,
      (maximum - splitVal) * 0.75);

  if (splitVal == maximum)
    splitVal = minimum;

  return true;
}

} // namespace tree
} // namespace mlpack

#endif // MLPACK_CORE_TREE_BINARY_SPACE_TREE_RP_TREE_MAX_SPLIT_IMPL_HPP
