/**
 * @file methods/decision_tree/splits/best_binary_numeric_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of strategy that finds the best binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_SPLITS_BEST_BINARY_NUMERIC_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_SPLITS_BEST_BINARY_NUMERIC_SPLIT_IMPL_HPP

namespace mlpack {

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
    totalWeight = accu(sortedWeights);
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
    if (data[sortedIndices[index - 1]] == data[sortedIndices[index]])
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

      // In some very extreme cases, floating-point inaccuracies can lead to the
      // split result being the upper bound, which is problematic for later as
      // all the child points will be sent to the left child.  If this happens,
      // bump it down incrementally.
      if (splitInfo[0] == data[sortedIndices[index]])
      {
        splitInfo[0] = std::nexttoward(splitInfo[0],
            data[sortedIndices[index - 1]]);
      }

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

      // In some very extreme cases, floating-point inaccuracies can lead to the
      // split result being the upper bound, which is problematic for later as
      // all the child points will be sent to the left child.  If this happens,
      // bump it down incrementally.
      if (splitInfo[0] == data[sortedIndices[index]])
      {
        splitInfo[0] = std::nexttoward(splitInfo[0],
            data[sortedIndices[index - 1]]);
      }
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
template<bool UseWeights, typename VecType, typename ResponsesType,
         typename WeightVecType>
std::enable_if_t<
    !HasOptimizedBinarySplitForms<FitnessFunction, UseWeights>::value,
    double>
BestBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const ResponsesType& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */,
    FitnessFunction& fitnessFunction)
{
  using RType = typename ResponsesType::elem_type;
  using WType = typename WeightVecType::elem_type;

  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimumLeafSize * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::Row<RType> sortedResponses(responses.n_elem);
  arma::Row<WType> sortedWeights;
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

  WType totalWeight = 0.0;
  WType totalLeftWeight = 0.0;
  WType totalRightWeight = 0.0;

  if (UseWeights)
  {
    totalWeight = accu(sortedWeights);
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
    const double leftGain = fitnessFunction.template
        Evaluate<UseWeights>(sortedResponses, sortedWeights, 0, index);
    const double rightGain = fitnessFunction.template
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
      splitInfo.set_size(1);
      splitInfo[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;

      // In some very extreme cases, floating-point inaccuracies can lead to the
      // split result being the upper bound, which is problematic for later as
      // all the child points will be sent to the left child.  If this happens,
      // bump it down incrementally.
      if (splitInfo[0] == data[sortedIndices[index]])
        splitInfo[0] = std::nexttoward(
            splitInfo[0], data[sortedIndices[index - 1]]);
      return gain;
    }
     if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      splitInfo.set_size(1);
      splitInfo[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      improved = true;

      // In some very extreme cases, floating-point inaccuracies can lead to the
      // split result being the upper bound, which is problematic for later as
      // all the child points will be sent to the left child.  If this happens,
      // bump it down incrementally.
      if (splitInfo[0] == data[sortedIndices[index]])
        splitInfo[0] = std::nexttoward(
            splitInfo[0], data[sortedIndices[index - 1]]);
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

// Optimized version for any fitness function that implements
// BinaryScanInitialize(), BinaryStep() and BinaryGains() functions.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename ResponsesType,
         typename WeightVecType>
std::enable_if_t<
    HasOptimizedBinarySplitForms<FitnessFunction, UseWeights>::value,
    double>
BestBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const ResponsesType& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */,
    FitnessFunction& fitnessFunction)
{
  using RType = typename ResponsesType::elem_type;
  using WType = typename WeightVecType::elem_type;

  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimumLeafSize * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  // Next, sort the data.
  arma::uvec sortedIndices = arma::sort_index(data);
  arma::Row<RType> sortedResponses(responses.n_elem);
  arma::Row<WType> sortedWeights;
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

  WType totalWeight = 0.0;
  WType leftChildWeight = 0.0;
  WType rightChildWeight = 0.0;

  if (UseWeights)
  {
    totalWeight = accu(sortedWeights);
    bestFoundGain *= totalWeight;

    for (size_t i = 0; i < minimum - 1; ++i)
      leftChildWeight += sortedWeights[i];

    for (size_t i = minimum - 1; i < data.n_elem; ++i)
      rightChildWeight += sortedWeights[i];
  }
  else
  {
    bestFoundGain *= data.n_elem;
  }

  // Initialize and precompute various statistics to efficiently compute gain
  // values for all possible splits.
  fitnessFunction.template BinaryScanInitialize<UseWeights>(sortedResponses,
      sortedWeights, minimum);

  // Loop through all possible split points, choosing the best one.
  for (size_t index = minimum; index < data.n_elem - minimum + 1; ++index)
  {
    if (UseWeights)
    {
      leftChildWeight += sortedWeights[index - 1];
      rightChildWeight -= sortedWeights[index - 1];
    }

    // Steps through the current index and updates the cached data.
    fitnessFunction.template BinaryStep<UseWeights>(sortedResponses,
        sortedWeights, index - 1);

    // Make sure that the value has changed.
    if (data[sortedIndices[index]] == data[sortedIndices[index - 1]])
      continue;

    // Calculate the gain for the left and right child.
    std::tuple<double, double> binaryGains = fitnessFunction.BinaryGains();
    const double leftGain = std::get<0>(binaryGains);
    const double rightGain = std::get<1>(binaryGains);

    double gain;
    if (UseWeights)
    {
      gain = leftChildWeight * leftGain + rightChildWeight * rightGain;
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
      splitInfo.set_size(1);
      splitInfo[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;

      // In some very extreme cases, floating-point inaccuracies can lead to the
      // split result being the upper bound, which is problematic for later as
      // all the child points will be sent to the left child.  If this happens,
      // bump it down incrementally.
      if (splitInfo[0] == data[sortedIndices[index]])
        splitInfo[0] = std::nexttoward(
            splitInfo[0], data[sortedIndices[index - 1]]);
      return gain;
    }
    if (gain > bestFoundGain)
    {
      // We still have a better split.
      bestFoundGain = gain;
      splitInfo.set_size(1);
      splitInfo[0] = (data[sortedIndices[index - 1]] +
          data[sortedIndices[index]]) / 2.0;
      improved = true;

      // In some very extreme cases, floating-point inaccuracies can lead to the
      // split result being the upper bound, which is problematic for later as
      // all the child points will be sent to the left child.  If this happens,
      // bump it down incrementally.
      if (splitInfo[0] == data[sortedIndices[index]])
        splitInfo[0] = std::nexttoward(
            splitInfo[0], data[sortedIndices[index - 1]]);
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
    const arma::vec& splitInfo,
    const AuxiliarySplitInfo& /* aux */)
{
  if (splitInfo.n_elem == 0)
    return SIZE_MAX;
  else if (point <= splitInfo[0])
    return 0; // Go left.
  else
    return 1; // Go right.
}

} // namespace mlpack

#endif
