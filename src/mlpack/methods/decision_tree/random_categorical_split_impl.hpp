/**
 * @file methods/decision_tree/random_categorical_split_impl.hpp
 * @author Adarsh Santoria
 *
 * Implementation of the RandomCategoricalSplit categorical split class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_RANDOM_CATEGORICAL_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_RANDOM_CATEGORICAL_SPLIT_IMPL_HPP

namespace mlpack {

// Overload used in classification.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename LabelsType,
         typename WeightVecType>
double RandomCategoricalSplit<FitnessFunction>::SplitIfBetter(
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
  const double epsilon = 1e-7; // Tolerance for floating-point errors.

  // Count the number of elements in each potential child.
  arma::vec categories(numCategories);
  arma::Col<size_t> counts(numCategories, arma::fill::zeros);
  size_t j = 0;
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    if(!counts[(size_t) data[i]])
      categories[j++]  = ((size_t) data[i]);
    counts[(size_t) data[i]]++;
  }

  arma::vec newCategories = arma::shuffle(categories);
  size_t randomPivot = Random(1, j);

  double totalCounts = 0;
  for (size_t i = 0; i < numCategories; ++i)
  {
    if(i >= randomPivot) counts[newCategories[i]] = 0;
    else
    {
      totalCounts += counts[newCategories[i]];
      // If each child will have the minimum number of points in it, we can split.
      // Otherwise we can't.
      if (counts[newCategories[i]] < minimumLeafSize)
        return DBL_MAX;
    }
  }
  
  // If we are using weighted training, split the weights for each child too.
  arma::vec childWeightSums;
  double sumWeight = 0.0;
  if (UseWeights)
    childWeightSums.zeros(numCategories);

  // Calculate the gain of the split.  First we have to calculate the labels
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories, arma::fill::zeros);
  std::vector<arma::Row<size_t>> childLabels(numCategories);
  std::vector<arma::Row<double>> childWeights(numCategories);

  for (size_t i = 0; i < randomPivot; ++i)
  {
    // Labels and weights should have same length.
    childLabels[newCategories[i]].zeros(counts[newCategories[i]]);
    if (UseWeights)
      childWeights[newCategories[i]].zeros(counts[newCategories[i]]);
  }

  // Extract labels for each child.
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    const size_t category = (size_t)data[i];

    if(!counts[category]) continue;

    if (UseWeights)
    {
      childWeights[category][childPositions[category]] = weights[i];
      childWeightSums[category] += weights[i];
      sumWeight += weights[i];
    }
    childLabels[category][childPositions[category]++] = labels[i];
  }

  double overallGain = 0.0;
  for (size_t i = 0; i < randomPivot; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = UseWeights ?
        double(childWeightSums[newCategories[i]]) / sumWeight :
        double(counts[newCategories[i]]) / double(totalCounts);
    const double childGain = FitnessFunction::template Evaluate<UseWeights>(
        childLabels[newCategories[i]], numClasses, childWeights[newCategories[i]]);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain + minimumGainSplit + epsilon)
  {
    // This is better, so store it in splitInfo and return.
    splitInfo.set_size(1);
    splitInfo[0] = randomPivot;
    return overallGain;
  }

  // Otherwise there was no improvement.
  return DBL_MAX;
}

// Overload used in regression.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename ResponsesType,
         typename WeightVecType>
double RandomCategoricalSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const size_t numCategories,
    const ResponsesType& responses,
    const WeightVecType& weights,
    const size_t minimumLeafSize,
    const double minimumGainSplit,
    double& splitInfo,
    AuxiliarySplitInfo& /* aux */,
    FitnessFunction& fitnessFunction)
{
  const double epsilon = 1e-7; // Tolerance for floating-point errors.

  // Count the number of elements in each potential child.
  arma::vec categories(numCategories);
  arma::Col<size_t> counts(numCategories, arma::fill::zeros);
  size_t j = 0;
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    if(!counts[(size_t) data[i]])
      categories[j++]  = ((size_t) data[i]);
    counts[(size_t) data[i]]++;
  }

  arma::vec newCategories = arma::shuffle(categories);
  size_t randomPivot = Random(1, j);

  double totalCounts = 0;
  for (size_t i = 0; i < numCategories; ++i)
  {
    if(i >= randomPivot) counts[newCategories[i]] = 0;
    else
    {
      totalCounts += counts[newCategories[i]];
      // If each child will have the minimum number of points in it, we can split.
      // Otherwise we can't.
      if (counts[newCategories[i]] < minimumLeafSize)
        return DBL_MAX;
    }
  }
  
  // If we are using weighted training, split the weights for each child too.
  arma::vec childWeightSums;
  double sumWeight = 0.0;
  if (UseWeights)
    childWeightSums.zeros(numCategories);

  // Calculate the gain of the split.  First we have to calculate the responses
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories, arma::fill::zeros);
  std::vector<arma::Row<size_t>> childResponses(numCategories);
  std::vector<arma::Row<double>> childWeights(numCategories);

  for (size_t i = 0; i < randomPivot; ++i)
  {
    // Labels and weights should have same length.
    childResponses[i].zeros(counts[newCategories[i]]);
    if (UseWeights)
      childWeights[i].zeros(counts[newCategories[i]]);
  }

  // Extract labels for each child.
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    const size_t category = (size_t)data[i];

    if(!counts[category]) continue;

    if (UseWeights)
    {
      childWeights[category][childPositions[category]] = weights[i];
      childWeightSums[category] += weights[i];
      sumWeight += weights[i];
    }
    childResponses[category][childPositions[category]++] = responses[i];
  }

  double overallGain = 0.0;
  for (size_t i = 0; i < randomPivot; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = UseWeights ?
        double(childWeightSums[newCategories[i]]) / sumWeight :
        double(counts[newCategories[i]]) / double(totalCounts);
    const double childGain = fitnessFunction.template Evaluate<UseWeights>(
        childResponses[newCategories[i]], childWeights[newCategories[i]]);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain + minimumGainSplit + epsilon)
  {
    // This is better, so store it in splitInfo and return.
    splitInfo = randomPivot;
    return overallGain;
  }

  // Otherwise there was no improvement.
  return DBL_MAX;
}

template<typename FitnessFunction>
size_t RandomCategoricalSplit<FitnessFunction>::NumChildren(
    const double& splitInfo,
    const AuxiliarySplitInfo& /* aux */)
{
  return (size_t) splitInfo;
}

template<typename FitnessFunction>
template<typename ElemType>
size_t RandomCategoricalSplit<FitnessFunction>::CalculateDirection(
    const ElemType& point,
    const double& /* splitInfo */,
    const AuxiliarySplitInfo& /* aux */)
{
  return (size_t) point;
}

} // namespace mlpack

#endif
