/**
 * @file methods/decision_tree/splits/random_binary_numeric_split_impl.hpp
 * @author Rishabh Garg
 *
 * Implementation of strategy that finds the random binary numeric split.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_SPLITS_RANDOM_BINARY_NUMERIC_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_SPLITS_RANDOM_BINARY_NUMERIC_SPLIT_IMPL_HPP

#include <mlpack/core/math/random.hpp>

namespace mlpack {

// Overload used for classification.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename WeightVecType>
double RandomBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */,
    const bool splitIfBetterGain)
{
  double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
  // Forcing a minimum leaf size of 1 (empty children don't make sense).
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);

  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimum * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  typename VecType::elem_type maxValue = max(data);
  typename VecType::elem_type minValue = min(data);

  // Sanity check: if the maximum element is the same as the minimum, we
  // can't split in this dimension.
  if (maxValue == minValue)
    return DBL_MAX;

  // Picking a random pivot to split the dimension.
  double randomPivot = Random(minValue, maxValue);

  // We need to count the number of points for each class.
  arma::Mat<size_t> classCounts;
  arma::mat classWeightSums;
  double totalWeight = 0.0;
  double totalLeftWeight = 0.0;
  double totalRightWeight = 0.0;
  size_t leftLeafSize = 0;
  size_t rightLeafSize = 0;
  if (UseWeights)
  {
    classWeightSums.zeros(numClasses, 2);
    totalWeight = accu(weights);
    bestFoundGain *= totalWeight;

    for (size_t i = 0; i < data.n_elem; ++i)
    {
      if (data(i) < randomPivot)
      {
        ++leftLeafSize;
        classWeightSums(labels(i), 0) += weights(i);
        totalLeftWeight += weights(i);
      }
      else
      {
        ++rightLeafSize;
        classWeightSums(labels(i), 1) += weights(i);
        totalRightWeight += weights(i);
      }
    }
  }
  else
  {
    classCounts.zeros(numClasses, 2);
    bestFoundGain *= data.n_elem;

    for (size_t i = 0; i < data.n_elem; ++i)
    {
      if (data(i) < randomPivot)
      {
        ++leftLeafSize;
        ++classCounts(labels(i), 0);
      }
      else
      {
        ++rightLeafSize;
        ++classCounts(labels(i), 1);
      }
    }
  }

  // Calculate the gain for the left and right child.  Only use weights if
  // needed.
  const double leftGain = UseWeights ?
      FitnessFunction::template EvaluatePtr<true>(classWeightSums.colptr(0),
          numClasses, totalLeftWeight) :
      FitnessFunction::template EvaluatePtr<false>(classCounts.colptr(0),
          numClasses, leftLeafSize);
  const double rightGain = UseWeights ?
      FitnessFunction::template EvaluatePtr<true>(classWeightSums.colptr(1),
          numClasses, totalRightWeight) :
      FitnessFunction::template EvaluatePtr<false>(classCounts.colptr(1),
          numClasses, rightLeafSize);

  double gain;
  if (UseWeights)
    gain = totalLeftWeight * leftGain + totalRightWeight * rightGain;
  else
    // Calculate the gain at this split point.
    gain = double(leftLeafSize) * leftGain + double(rightLeafSize) * rightGain;

  if (gain < bestFoundGain && splitIfBetterGain)
    return DBL_MAX;

  splitInfo.set_size(1);
  splitInfo[0] = randomPivot;

  if (UseWeights)
    gain /= totalWeight;
  else
    gain /= labels.n_elem;

  return gain;
}

// Overload used for regression.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename WeightVecType>
double RandomBinaryNumericSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const arma::rowvec& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */,
    FitnessFunction& fitnessFunction,
    const bool splitIfBetterGain)
{
  double bestFoundGain = std::min(bestGain + minimumGainSplit, 0.0);
  // Forcing a minimum leaf size of 1 (empty children don't make sense).
  const size_t minimum = std::max(minimumLeafSize, (size_t) 1);

  // First sanity check: if we don't have enough points, we can't split.
  if (data.n_elem < (minimum * 2))
    return DBL_MAX;
  if (bestGain == 0.0)
    return DBL_MAX; // It can't be outperformed.

  typename VecType::elem_type maxValue = max(data);
  typename VecType::elem_type minValue = min(data);

  // Sanity check: if the maximum element is the same as the minimum, we
  // can't split in this dimension.
  if (maxValue == minValue)
    return DBL_MAX;

  double totalWeight = 0.0;
  double totalLeftWeight = 0.0;
  double totalRightWeight = 0.0;
  if (UseWeights)
  {
    totalWeight = accu(weights);
    bestFoundGain *= totalWeight;
  }
  else
  {
    bestFoundGain *= data.n_elem;
  }

  // Picking a random pivot to split the dimension.
  double randomPivot = Random(minValue, maxValue);

  // We need to count the number of points for each leaf.
  size_t leftLeafSize = 0;
  size_t rightLeafSize = 0;
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    if (UseWeights)
    {
      if (data[i] < randomPivot)
        totalLeftWeight += weights[i];
      else
        totalRightWeight += weights[i];
    }

    if (data[i] < randomPivot)
      ++leftLeafSize;
    else
      ++rightLeafSize;
  }

  // Splitting data to compute gain.
  arma::rowvec leftResponses(leftLeafSize), rightResponses(rightLeafSize);
  arma::rowvec leftWeights, rightWeights;
  if (UseWeights)
  {
    leftWeights.set_size(leftLeafSize);
    rightWeights.set_size(rightLeafSize);
  }

  size_t l = 0, r = 0;
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    if (UseWeights)
    {
      if (data[i] < randomPivot)
        leftWeights[l] = weights[i];
      else
        rightWeights[r] = weights[i];
    }
    if (data[i] < randomPivot)
      leftResponses[l++] = responses[i];
    else
      rightResponses[r++] = responses[i];
  }

  // Calculate the gain for the left and right child.
  const double leftGain = fitnessFunction.template
      Evaluate<UseWeights>(leftResponses, leftWeights, 0, leftLeafSize);
  const double rightGain = fitnessFunction.template
      Evaluate<UseWeights>(rightResponses, rightWeights, 0, rightLeafSize);

  // Calculate the gain at this split point.
  double gain;
  if (UseWeights)
    gain = totalLeftWeight * leftGain + totalRightWeight * rightGain;
  else
    gain = double(leftLeafSize) * leftGain + double(rightLeafSize) * rightGain;

  if (gain < bestFoundGain && splitIfBetterGain)
    return DBL_MAX;

  splitInfo.set_size(1);
  splitInfo[0] = randomPivot;

  if (UseWeights)
    gain /= totalWeight;
  else
    gain /= responses.n_elem;

  return gain;
}

template<typename FitnessFunction>
template<typename ElemType>
size_t RandomBinaryNumericSplit<FitnessFunction>::CalculateDirection(
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
