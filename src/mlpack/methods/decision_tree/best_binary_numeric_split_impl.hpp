/**
 * @file methods/decision_tree/best_binary_numeric_split_impl.hpp
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

// Overload used for classification.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename WeightVecType>
double BestBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */)
{
  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimumLeafSize * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::Row<size_t> sortedLabels(labels.n_elem);
  arma::rowvec sortedWeights;
  for (size_t i = 0; i < sortedLabels.n_elem; ++i)
    sortedLabels[i] = labels[sortedIndices[i]];

  // Sanity check: if the first element is the same as the last, we can't split
  // in this dimension.
  if (data[sortedIndices[0]] == data[sortedIndices[sortedIndices.n_elem - 1]])
    return DBL_MAX;

  // Only initialize if we are using weights.
  if (UseWeights)
  {
    sortedWeights.set_size(sortedLabels.n_elem);
    // The weights must keep the same order as the labels.
    for (size_t i = 0; i < sortedLabels.n_elem; ++i)
      sortedWeights[i] = weights[sortedIndices[i]];
  }

  // Loop through all possible split points, choosing the best one.  Also, force
  // a minimum leaf size of 1 (empty children don't make sense).
  double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
  bool improved = false;
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);

  // We need to count the number of points for each class.
  arma::Mat<size_t> classCounts;
  arma::mat classWeightSums;
  double totalWeight = 0.0;
  double totalLeftWeight = 0.0;
  double totalRightWeight = 0.0;
  if (UseWeights)
  {
    classWeightSums.zeros(numClasses, 2);
    totalWeight = arma::accu(sortedWeights);
    bestFoundGain *= totalWeight;

    // Initialize the counts.
    // These points have to be on the left.
    for (size_t i = 0; i < minimum - 1; ++i)
    {
      classWeightSums(sortedLabels[i], 0) += sortedWeights[i];
      totalLeftWeight += sortedWeights[i];
    }

    // These points have to be on the right.
    for (size_t i = minimum - 1; i < data.n_elem; ++i)
    {
      classWeightSums(sortedLabels[i], 1) += sortedWeights[i];
      totalRightWeight += sortedWeights[i];
    }
  }
  else
  {
    classCounts.zeros(numClasses, 2);
    bestFoundGain *= data.n_elem;

    // Initialize the counts.
    // These points have to be on the left.
    for (size_t i = 0; i < minimum - 1; ++i)
      ++classCounts(sortedLabels[i], 0);

    // These points have to be on the right.
    for (size_t i = minimum - 1; i < data.n_elem; ++i)
      ++classCounts(sortedLabels[i], 1);
  }

  for (size_t index = minimum; index < data.n_elem - minimum; ++index)
  {
    // Update class weight sums or counts.
    if (UseWeights)
    {
      classWeightSums(sortedLabels[index - 1], 1) -= sortedWeights[index - 1];
      classWeightSums(sortedLabels[index - 1], 0) += sortedWeights[index - 1];
      totalLeftWeight += sortedWeights[index - 1];
      totalRightWeight -= sortedWeights[index - 1];
    }
    else
    {
      --classCounts(sortedLabels[index - 1], 1);
      ++classCounts(sortedLabels[index - 1], 0);
    }

    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Calculate the gain for the left and right child.  Only use weights if
    // needed.
    const double leftGain = UseWeights ?
        FitnessFunction::template EvaluatePtr<true>(classWeightSums.colptr(0),
            numClasses, totalLeftWeight) :
        FitnessFunction::template EvaluatePtr<false>(classCounts.colptr(0),
            numClasses, index);
    const double rightGain = UseWeights ?
        FitnessFunction::template EvaluatePtr<true>(classWeightSums.colptr(1),
            numClasses, totalRightWeight) :
        FitnessFunction::template EvaluatePtr<false>(classCounts.colptr(1),
            numClasses, size_t(sortedLabels.n_elem - index));

    double gain;
    if (UseWeights)
    {
      gain = totalLeftWeight * leftGain + totalRightWeight * rightGain;
    }
    else
    {
      // Calculate the gain at this split point.
      gain = double(index) * leftGain +
          double(sortedLabels.n_elem - index) * rightGain;
    }

    // Corner case: is this the best possible split?
    if (gain >= 0.0)
    {
      // We can take a shortcut: no split will be better than this, so just
      // take this one. The actual split value will be halfway between the
      // value at index - 1 and index.
      splitInfo.set_size(1);
      splitInfo[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;

      return gain;
    }
    else if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      splitInfo.set_size(1);
      splitInfo[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      improved = true;
    }
  }

  // If we didn't improve, return the original gain exactly as we got it
  // (without introducing floating point errors).
  if (!improved)
    return DBL_MAX;

  if (UseWeights)
    bestFoundGain /= totalWeight;
  else
    bestFoundGain /= sortedLabels.n_elem;

  return bestFoundGain;
}

// Overload used for regression.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename WeightVecType>
double BestBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const arma::rowvec& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    double& splitInfo,
    AuxiliarySplitInfo& /* aux */)
{
  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimumLeafSize * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::rowvec sortedResponses(responses.n_elem);
  arma::rowvec sortedWeights;
  for (size_t i = 0; i < sortedResponses.n_elem; ++i)
    sortedResponses[i] = responses[sortedIndices[i]];

  // Sanity check: if the first element is the same as the last, we can't split
  // in this dimension.
  if (data[sortedIndices[0]] == data[sortedIndices[sortedIndices.n_elem - 1]])
    return DBL_MAX;

  // Only initialize if we are using weights.
  if (UseWeights)
  {
    sortedWeights.set_size(sortedResponses.n_elem);
    // The weights must keep the same order as the responses.
    for (size_t i = 0; i < sortedResponses.n_elem; ++i)
      sortedWeights[i] = weights[sortedIndices[i]];
  }

  double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
  bool improved = false;
  // Force a minimum leaf size of 1 (empty children don't make sense).
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);

  double totalWeight = 0.0;
  double totalLeftWeight = 0.0;
  double totalRightWeight = 0.0;

  if (UseWeights)
  {
    totalWeight = arma::accu(sortedWeights);
    bestFoundGain *= totalWeight;

    for (size_t i = 0; i < minimum - 1; ++i)
      totalLeftWeight += sortedWeights[i];

    for (size_t i = minimum - 1; i < data.n_elem; ++i)
      totalRightWeight += sortedWeights[i];
  }
  else
  {
    bestFoundGain *= data.n_elem;
  }

  // Loop through all possible split points, choosing the best one.
  for (size_t index = minimum; index < data.n_elem - minimum + 1; ++index)
  {
    if (UseWeights)
    {
      totalLeftWeight += sortedWeights[index - 1];
      totalRightWeight -= sortedWeights[index - 1];
    }
    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Calculate the gain for the left and right child.
    const double leftGain = FitnessFunction::template
        Evaluate<UseWeights>(sortedResponses, sortedWeights, 0, index);
    const double rightGain = FitnessFunction::template
        Evaluate<UseWeights>(sortedResponses, sortedWeights, index,
            responses.n_elem);

    double gain;
    if (UseWeights)
    {
      gain = totalLeftWeight * leftGain + totalRightWeight * rightGain;
    }
    else
    {
      // Calculate the gain at this split point.
      gain = double(index) * leftGain +
          double(sortedResponses.n_elem - index) * rightGain;
    }

    // Corner case: is this the best possible split?
    if (gain >= 0.0)
    {
      // We can take a shortcut: no split will be better than this, so just
      // take this one. The actual split value will be halfway between the
      // value at index - 1 and index.
      splitInfo = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;

      return gain;
    }
     if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      splitInfo = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      improved = true;
    }
  }

  // If we didn't improve, return the original gain exactly as we got it
  // (without introducing floating point errors).
  if (!improved)
    return DBL_MAX;

  if (UseWeights)
    bestFoundGain /= totalWeight;
  else
    bestFoundGain /= data.n_elem;

  return bestFoundGain;
}

// Optimized version when fitness function is MSEGain.
template<>
template<bool UseWeights, typename VecType, typename WeightVecType>
double BestBinaryNumericSplit<MSEGain>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const arma::rowvec& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    double& splitInfo,
    AuxiliarySplitInfo& /* aux */)
{
  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimumLeafSize * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::rowvec sortedResponses(responses.n_elem);
  arma::rowvec sortedWeights;
  for (size_t i = 0; i < sortedResponses.n_elem; ++i)
    sortedResponses[i] = responses[sortedIndices[i]];

  // Sanity check: if the first element is the same as the last, we can't split
  // in this dimension.
  if (data[sortedIndices[0]] == data[sortedIndices[sortedIndices.n_elem - 1]])
    return DBL_MAX;

  // Only initialize if we are using weights.
  if (UseWeights)
  {
    sortedWeights.set_size(sortedResponses.n_elem);
    // The weights must keep the same order as the responses.
    for (size_t i = 0; i < sortedResponses.n_elem; ++i)
      sortedWeights[i] = weights[sortedIndices[i]];
  }

  double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
  bool improved = false;
  // Force a minimum leaf size of 1 (empty children don't make sense).
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);

  double totalWeight = 0.0;
  double leftChildWeight = 0.0;
  double rightChildWeight = 0.0;
  double leftWeightedMean = 0.0;
  double rightWeightedMean = 0.0;
  double totalWeightedSumSquares = 0.0;
  arma::rowvec weightedSumSquares;

  double leftMean = 0.0;
  double rightMean = 0.0;
  size_t leftChildSize = 0;
  size_t rightChildSize = 0;
  double totalSumSquares = 0.0;
  arma::rowvec sumSquares;

  // Precomputing prefix sum of squares and prefix weighted sum of squares.
  // This will be used by MSEGain::Evaluate to efficiently compute gain
  // values for all possible splits.
  if (UseWeights)
  {
    totalWeight = arma::accu(sortedWeights);
    bestFoundGain *= totalWeight;

    weightedSumSquares.set_size(data.n_elem);
    // Stores the weighted sum of squares till the previous index.
    double prevWeightedSumSquares = 0.0;

    for (size_t i = 0; i < minimum - 1; ++i)
    {
      const double w = sortedWeights[i];
      const double x = sortedResponses[i];

      // Calculating initial weighted mean of responses for the left child.
      leftChildWeight += w;
      leftWeightedMean += w * x;
      weightedSumSquares[i] = prevWeightedSumSquares + w * x * x;
      prevWeightedSumSquares += w * x * x;
    }
    if (leftChildWeight > 1e-9)
      leftWeightedMean /= leftChildWeight;

    for (size_t i = minimum - 1; i < data.n_elem; ++i)
    {
      const double w = sortedWeights[i];
      const double x = sortedResponses[i];

      // Calculating initial weighted mean of responses for the right child.
      rightChildWeight += w;
      rightWeightedMean += w * x;
      weightedSumSquares[i] = prevWeightedSumSquares + w * x * x;
      prevWeightedSumSquares += w * x * x;
    }
    if (rightChildWeight > 1e-9)
      rightWeightedMean /= rightChildWeight;

    totalWeightedSumSquares = prevWeightedSumSquares;
  }
  else
  {
    bestFoundGain *= data.n_elem;

    sumSquares.set_size(data.n_elem);
    // Stores the sum of squares till the previous index.
    double prevSumSquares = 0.0;

    for (size_t i = 0; i < minimum - 1; ++i)
    {
      const double x = sortedResponses[i];

      // Calculating the initial mean of responses for the left child.
      ++leftChildSize;
      leftMean += x;
      sumSquares[i] = prevSumSquares + x * x;
      prevSumSquares += x * x;
    }
    if (leftChildSize)
      leftMean /= (double) leftChildSize;

    for (size_t i = minimum - 1; i < data.n_elem; ++i)
    {
      const double x = sortedResponses[i];

      // Calculating the initial mean of responses for the right child.
      rightChildSize++;
      rightMean += x;
      sumSquares[i] = prevSumSquares + x * x;
      prevSumSquares += x * x;
    }
    if (rightChildSize)
      rightMean /= (double) rightChildSize;

    totalSumSquares = prevSumSquares;
  }

  // Loop through all possible split points, choosing the best one.
  for (size_t index = minimum; index < data.n_elem - minimum + 1; ++index)
  {
    if (UseWeights)
    {
      // Updating the weighted mean for both childs for each index.
      const double w = sortedWeights[index - 1];
      const double x = sortedResponses[index - 1];
      leftWeightedMean = (leftWeightedMean * leftChildWeight + w * x)
          / (leftChildWeight + w);
      leftChildWeight += w;

      rightWeightedMean = (rightWeightedMean * rightChildWeight - w * x)
          / (rightChildWeight - w);
      rightChildWeight -= w;
    }
    else
    {
      // Updating the mean for both childs for each index.
      const double x = sortedResponses[index - 1];
      leftMean = (leftMean * (double) leftChildSize + x) /
          (double) (leftChildSize + 1);
      ++leftChildSize;

      rightMean = (rightMean * (double) rightChildSize - x) /
          (double) (rightChildSize - 1);
      --rightChildSize;
    }

    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Calculate the gain for the left and right child.
    const double leftGain = UseWeights ?
        MSEGain::Evaluate(weightedSumSquares[index - 1],
            leftWeightedMean, leftChildWeight) :
        MSEGain::Evaluate(sumSquares[index - 1], leftMean, leftChildSize);
    const double rightGain = UseWeights ?
        MSEGain::Evaluate(
            totalWeightedSumSquares - weightedSumSquares[index - 1],
            rightWeightedMean, rightChildWeight) :
        MSEGain::Evaluate(totalSumSquares - sumSquares[index - 1],
            rightMean, rightChildSize);

    double gain;
    if (UseWeights)
    {
      gain = leftChildWeight * leftGain + rightChildWeight * rightGain;
    }
    else
    {
      // Calculate the gain at this split point.
      gain = double(leftChildSize) * leftGain +
          double(rightChildSize) * rightGain;
    }

    // Corner case: is this the best possible split?
    if (gain >= 0.0)
    {
      // We can take a shortcut: no split will be better than this, so just
      // take this one. The actual split value will be halfway between the
      // value at index - 1 and index.
      splitInfo = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;

      return gain;
    }
     if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      splitInfo = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      improved = true;
    }
  }
  // If we didn't improve, return the original gain exactly as we got it
  // (without introducing floating point errors).
  if (!improved)
    return DBL_MAX;

  if (UseWeights)
    bestFoundGain /= totalWeight;
  else
    bestFoundGain /= data.n_elem;

  return bestFoundGain;
}

template<typename FitnessFunction>
template<typename ElemType>
size_t BestBinaryNumericSplit<FitnessFunction>::CalculateDirection(
    const ElemType& point,
    const double& splitInfo,
    const AuxiliarySplitInfo& /* aux */)
{
  if (point <= splitInfo)
    return 0; // Go left.
  else
    return 1; // Go right.
}

} // namespace tree
} // namespace mlpack

#endif
