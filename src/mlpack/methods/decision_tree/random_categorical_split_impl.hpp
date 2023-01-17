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
  arma::Col<size_t> counts(numCategories, arma::fill::zeros);
  for (size_t i = 0; i < data.n_elem; ++i)
    counts[(size_t) data[i]]++;
  
  // Calculate the gain of the split.  First we have to calculate the labels
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories, arma::fill::zeros);
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
    const size_t category = (size_t)data[i];

    if (UseWeights)
      childWeights[category][childPositions[category]] = weights[i];
    childLabels[category][childPositions[category]++] = labels[i];
  }

  // Shuffling the childs
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
    numCategories - 1, numCategories));

  // Picking two random Pivot to make 2 subsets and take their union
  size_t randomPivot1 = Random(1, numCategories - 1);
  size_t randomPivot2 = Random(randomPivot1, numCategories - 1);

  // Making new Variables containing childs of union of the subsets
  arma::Col<size_t> newCounts(randomPivot1 + numCategories - randomPivot2);
  std::vector<arma::Row<size_t>> 
    newLabels(randomPivot1 + numCategories - randomPivot2);
  std::vector<arma::Row<double>> 
    newWeights(randomPivot1 + numCategories - randomPivot2);
  arma::vec newWeightSums(randomPivot1 + numCategories - randomPivot2);

  double sumWeight = 0.0;
  size_t totalCounts = 0;
  for (size_t i = 0, j = 0; i < numCategories; ++i){
    if(i < randomPivot1 || i >= randomPivot2)
    {
      if (UseWeights)
      {
        newWeights[j] = childWeights[ordering[i]];
        newWeightSums[j] = accu(childWeights[ordering[i]]);
        sumWeight += newWeightSums[j];
      }
      newCounts[j] = counts[ordering[i]];
      totalCounts += counts[ordering[i]];
      newLabels[j++] = childLabels[ordering[i]];
    }
  }

  // If each child will have the minimum number of points in it, we can split.
  // Otherwise we can't.
  if (arma::min(newCounts) < minimumLeafSize)
    return DBL_MAX;

  double overallGain = 0.0;
  for (size_t i = 0; i < newCounts.n_elem; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = UseWeights ?
        double(newWeightSums[i]) / sumWeight :
        double(newCounts[i]) / double(totalCounts);
    const double childGain = FitnessFunction::template Evaluate<UseWeights>(
        newLabels[i], numClasses, newWeights[i]);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain + minimumGainSplit + epsilon)
  {
    // This is better, so store it in splitInfo and return.
    splitInfo.set_size(1);
    splitInfo[0] = randomPivot1 + numCategories - randomPivot2;
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
  arma::Col<size_t> counts(numCategories, arma::fill::zeros);
  for (size_t i = 0; i < data.n_elem; ++i)
    counts[(size_t) data[i]]++;
  
  // Calculate the gain of the split.  First we have to calculate the responses
  // that would be assigned to each child.
  arma::uvec childPositions(numCategories, arma::fill::zeros);
  std::vector<arma::Row<size_t>> childResponses(numCategories);
  std::vector<arma::Row<double>> childWeights(numCategories);

  for (size_t i = 0; i < numCategories; ++i)
  {
    // Responses and weights should have same length.
    childResponses[i].zeros(counts[i]);
    if (UseWeights)
      childWeights[i].zeros(counts[i]);
  }

  // Extract responses for each child.
  for (size_t i = 0; i < data.n_elem; ++i)
  {
    const size_t category = (size_t)data[i];

    if (UseWeights)
      childWeights[category][childPositions[category]] = weights[i];
    childResponses[category][childPositions[category]++] = responses[i];
  }

  // Shuffling the childs
  arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
    numCategories - 1, numCategories));

  // Picking two random Pivot to make 2 subsets and take their union
  size_t randomPivot1 = Random(1, numCategories - 1);
  size_t randomPivot2 = Random(randomPivot1,numCategories - 1);

  // Making new Variables containing childs of union of the subsets
  arma::Col<size_t> newCounts(randomPivot1 + numCategories - randomPivot2);
  std::vector<arma::Row<size_t>> 
    newResponses(randomPivot1 + numCategories - randomPivot2);
  std::vector<arma::Row<double>> 
    newWeights(randomPivot1 + numCategories - randomPivot2);
  arma::vec newWeightSums(randomPivot1 + numCategories - randomPivot2);

  double sumWeight = 0.0;
  size_t totalCounts = 0;
  for (size_t i = 0, j = 0; i < numCategories; ++i){
    if(i < randomPivot1 || i >= randomPivot2)
    {
      if (UseWeights)
      {
        newWeights[j] = childWeights[ordering[i]];
        newWeightSums[j] = accu(childWeights[ordering[i]]);
        sumWeight += newWeightSums[j];
      }
      newCounts[j] = counts[ordering[i]];
      totalCounts += counts[ordering[i]];
      newResponses[j++] = childResponses[ordering[i]];
    }
  }

  // If each child will have the minimum number of points in it, we can split.
  // Otherwise we can't.
  if (arma::min(newCounts) < minimumLeafSize)
    return DBL_MAX;

  double overallGain = 0.0;
  for (size_t i = 0; i < newCounts.n_elem; ++i)
  {
    // Calculate the gain of this child.
    const double childPct = UseWeights ?
        double(newWeightSums[i]) / sumWeight :
        double(newCounts[i]) / double(totalCounts);
    const double childGain = FitnessFunction::template Evaluate<UseWeights>(
        newResponses[i], newWeights[i]);

    overallGain += childPct * childGain;
  }

  if (overallGain > bestGain + minimumGainSplit + epsilon)
  {
    // This is better, so store it in splitInfo and return.
    splitInfo = randomPivot1 + numCategories - randomPivot2;
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
