/**
 * @file best_binary_numeric_split_impl.hpl
 * @author Ryan Curtin
 *
 * Implementation of strategy that finds the best binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_BEST_BINARY_NUMERIC_SPLIT_IMPL_HPP

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
template<typename VecType>
double BestBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t minimumLeafSize,
    arma::Col<typename VecType::elem_type>& classProbabilities,
    AuxiliarySplitInfo<typename VecType::elem_type>& /* aux */)
{
  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimumLeafSize * 2))
    return bestGain;

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::Row<size_t> sortedLabels(labels.n_elem);
  for (size_t i = 0; i < sortedLabels.n_elem; ++i)
    sortedLabels[sortedIndices[i]] = labels[i];

  // Loop through all possible split points, choosing the best one.  Also, force
  // a minimum leaf size of 1 (empty children don't make sense).
  double bestFoundGain = bestGain;
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);
  for (size_t index = minimum; index < data.n_elem - (minimum - 1); ++index)
  {
    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Calculate the gain for the left and right child.
    const double leftGain = FitnessFunction::Evaluate(sortedLabels.subvec(0,
        index - 1), numClasses);
    const double rightGain = FitnessFunction::Evaluate(sortedLabels.subvec(
        index, sortedLabels.n_elem - 1), numClasses);

    // Calculate the fraction of points in the left and right children.
    const double leftRatio = double(index) / double(sortedLabels.n_elem);
    const double rightRatio = 1.0 - leftRatio;

    // Calculate the gain at this split point.
    const double gain = leftRatio * leftGain + rightRatio * rightGain;

    // Corner case: is this the best possible split?
    if (gain == 0.0)
    {
      // We can take a shortcut: no split will be better than this, so just take
      // this one.
      classProbabilities.set_size(1);
      // The actual split value will be halfway between the value at index - 1
      // and index.
      classProbabilities[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      return gain;
    }
    else if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      classProbabilities.set_size(1);
      classProbabilities[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
    }
  }

  return bestFoundGain;
}

template<typename FitnessFunction>
template<typename ElemType>
size_t BestBinaryNumericSplit<FitnessFunction>::CalculateDirection(
    const ElemType& point,
    const arma::Col<ElemType>& classProbabilities,
    const AuxiliarySplitInfo<ElemType>& /* aux */)
{
  if (point <= classProbabilities[0])
    return 0; // Go left.
  else
    return 1; // Go right.
}

} // namespace tree
} // namespace mlpack

#endif
