/**
 * @file methods/decision_tree/splits/all_categorical_split_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the AllCategoricalSplit categorical split class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_SPLITS_ALL_CATEGORICAL_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_SPLITS_ALL_CATEGORICAL_SPLIT_IMPL_HPP

namespace mlpack {

// Overload used in classification.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename LabelsType,
         typename WeightVecType>
double AllCategoricalSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const size_t numCategories,
    const LabelsType& labels,
    const size_t numClasses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */)
{
  // Count the number of elements in each potential child.
  const double epsilon = 1e-7; // Tolerance for floating-point errors.
  arma::Col<size_t> counts(numCategories);

  // If we are using weighted training, split the weights for each child too.
  arma::vec childWeightSums;
  double sumWeight = 0.0;
  if (UseWeights)
    childWeightSums.zeros(numCategories);

  for (size_t i = 0; i < data.n_elem; ++i)
  {
    counts[(size_t) data[i]]++;

    if (UseWeights)
    {
      childWeightSums[(size_t) data[i]] += weights[i];
      sumWeight += weights[i];
    }
  }

  // If each child will have the minimum number of points in it, we can split.
  // Otherwise we can't.
  if (min(counts) < minimumLeafSize)
    return DBL_MAX;

  // Calculate the gain of the split.  First we have to calculate the labels
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories);
  std::vector<arma::Row<size_t>> childLabels(numCategories);
  std::vector<arma::Row<double>> childWeights(numCategories);

  for (size_t i = 0; i < numCategories; ++i)
  {
    // Labels and weights should have same length.
    childLabels[i].zeros(counts[i]);
    if (UseWeights)
      childWeights[i].zeros(counts[i]);
  }

  // Extract labels for each child.
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    const size_t category = (size_t) data[i];

    if (UseWeights)
    {
      childLabels[category][childPositions[category]] = labels[i];
      childWeights[category][childPositions[category]++] = weights[i];
    }
    else
    {
      childLabels[category][childPositions[category]++] = labels[i];
    }
  }

  double overallGain = 0.0;
  for (size_t i = 0; i < counts.n_elem; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = UseWeights ?
        double(childWeightSums[i]) / sumWeight :
        double(counts[i]) / double(data.n_elem);
    const double childGain = FitnessFunction::template Evaluate<UseWeights>(
        childLabels[i], numClasses, childWeights[i]);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain + minimumGainSplit + epsilon)
  {
    // This is better, so store it in splitInfo and return.
    splitInfo.set_size(1);
    splitInfo[0] = numCategories;
    return overallGain;
  }

  // Otherwise there was no improvement.
  return DBL_MAX;
}

// Overload used in regression.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename ResponsesType,
         typename WeightVecType>
double AllCategoricalSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const size_t numCategories,
    const ResponsesType& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& /* aux */,
    FitnessFunction& fitnessFunction)
{
  // Count the number of elements in each potential child.
  const double epsilon = 1e-7; // Tolerance for floating-point errors.
  arma::Col<size_t> counts(numCategories);

  // If we are using weighted training, split the weights for each child too.
  arma::vec childWeightSums;
  double sumWeight = 0.0;
  if (UseWeights)
    childWeightSums.zeros(numCategories);

  for (size_t i = 0; i < data.n_elem; ++i)
  {
    counts[(size_t) data[i]]++;

    if (UseWeights)
    {
      childWeightSums[(size_t) data[i]] += weights[i];
      sumWeight += weights[i];
    }
  }

  // If each child will have the minimum number of points in it, we can split.
  // Otherwise we can't.
  if (min(counts) < minimumLeafSize)
    return DBL_MAX;

  // Calculate the gain of the split.  First we have to calculate the labels
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories);
  std::vector<arma::rowvec> childResponses(numCategories);
  std::vector<arma::rowvec> childWeights(numCategories);

  for (size_t i = 0; i < numCategories; ++i)
  {
    // Responses and weights should have same length.
    childResponses[i].zeros(counts[i]);
    if (UseWeights)
      childWeights[i].zeros(counts[i]);
  }

  // Extract labels for each child.
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    const size_t category = (size_t) data[i];

    if (UseWeights)
    {
      childResponses[category][childPositions[category]] = responses[i];
      childWeights[category][childPositions[category]++] = weights[i];
    }
    else
    {
      childResponses[category][childPositions[category]++] = responses[i];
    }
  }

  double overallGain = 0.0;
  for (size_t i = 0; i < counts.n_elem; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = UseWeights ?
        double(childWeightSums[i]) / sumWeight :
        double(counts[i]) / double(data.n_elem);
    const double childGain = fitnessFunction.template Evaluate<UseWeights>(
        childResponses[i], childWeights[i]);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain + minimumGainSplit + epsilon)
  {
    // This is better, so store it in splitInfo and return.
    splitInfo.set_size(1);
    splitInfo[0] = numCategories;
    return overallGain;
  }

  // Otherwise there was no improvement.
  return DBL_MAX;
}

template<typename FitnessFunction>
size_t AllCategoricalSplit<FitnessFunction>::NumChildren(
    const arma::vec& splitInfo,
    const AuxiliarySplitInfo& /* aux */)
{
  return splitInfo.n_elem == 0 ? 0 : (size_t) splitInfo[0];
}

template<typename FitnessFunction>
template<typename ElemType>
size_t AllCategoricalSplit<FitnessFunction>::CalculateDirection(
    const ElemType& point,
    const arma::vec& splitInfo,
    const AuxiliarySplitInfo& /* aux */)
{
  return splitInfo.n_elem == 0 ? SIZE_MAX : (size_t) point;
}

} // namespace mlpack

#endif
