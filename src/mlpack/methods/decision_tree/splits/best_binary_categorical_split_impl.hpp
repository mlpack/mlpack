/**
 * @file methods/decision_tree/splits/all_categorical_split_impl.hpp
 * @author Nikolay Apanasov (nikolay@apanasov.org)
 *
 * Implementation of the BestBinaryCategoricalSplit categorical split class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_DECISION_TREE_BEST_BINARY_CATEGORICAL_SPLIT_IMPL_HPP
#define MLPACK_METHODS_DECISION_TREE_BEST_BINARY_CATEGORICAL_SPLIT_IMPL_HPP

// In case it hasn't been included yet.
#include "best_binary_categorical_split.hpp"

namespace mlpack {

// Overload used in classification.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename LabelsType,
         typename WeightVecType>
double BestBinaryCategoricalSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const size_t numCategories,
    const LabelsType& labels,
    const size_t numClasses,
    const WeightVecType& weights,
    const size_t minLeafSize,
    const double minGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& aux)
{
  const size_t n = data.n_elem;
  double bestFoundGain = std::min(bestGain + minGainSplit, 0.0);
  bool improved = false;

  // We are too small to split.
  if (n < (minLeafSize * 2))
    return DBL_MAX;

  // Binary classification
  if (numClasses == 2)
  {
      // Order the categories of variable vₖ by their proportion in class C₁
      // and map each categorical vₖ to its categorical rank
      arma::umat categoryCounts(numCategories, 2);
      arma::vec categoryP(numCategories);
      size_t totalCount;

      for (size_t i = 0; i < n; ++i)
        ++categoryCounts(data[i], labels[i]);
      for (size_t i = 0; i < numCategories; ++i)
      {
        totalCount = (categoryCounts(i, 0) + categoryCounts(i, 1));
        categoryP[i] = totalCount == 0 ? 0 : (categoryCounts(i, 1)
            / (categoryCounts(i, 0) + categoryCounts(i, 1)));
      }
      arma::uvec sortedCategories = sort_index(categoryP);
      arma::uvec categoryRank(numCategories);
      for (size_t i = 0; i < numCategories; i++)
        categoryRank[sortedCategories[i]] = i;

      arma::uvec transformedData(n);
      for (size_t i = 0; i < n; ++i)
         transformedData[i] = categoryRank[data[i]];

      // Split the transformed vₖ as a numeric type.
      bestFoundGain = NumericSplit::template SplitIfBetter<UseWeights>(
          bestFoundGain,
          transformedData,
          labels,
          numClasses,
          weights,
          minLeafSize,
          minGainSplit,
          splitInfo,
          (NumericAux &) aux);
      improved = bestFoundGain != DBL_MAX;

      // This split is better: store the set membership in splitInfo
      // and return. Thus splitInfo is a vector of size Q, where Q is
      // the number of categories, and splitInfo[k] is zero if category
      // k is assigned to the left child, and otherwise it is one if k
      // is assigned to the right.
      if (improved)
      {
        const size_t splitIndex = (size_t) std::floor((double) splitInfo[0]);
        splitInfo.set_size(numCategories);
        for (size_t c : sortedCategories.subvec(0, splitIndex))
          splitInfo[c] = 0;
        for (size_t c : sortedCategories.subvec(splitIndex + 1,
                                                numCategories - 1))
          splitInfo[c] = 1;
      }
  }
  // Multi-class classification -- Brute force search through all the
  // 2ʲ possible partitions (Gₗ, Gᵣ) of the categories C₀, ..., Cⱼ,
  // assigning samples with vₖ ∈ Gₗ to left tree Tₗ and those with vₖ ∈ Gᵣ
  // to right tree Tᵣ.
  else
  {
    // A map from category Cⱼ to the samples whose categorical value
    // for variable vₖ is Cⱼ. The jth column corresponds to Cⱼ.
    arma::SpMat<short> categorySamples(n, numCategories);
    for (size_t i = 0; i < n; ++i)
      categorySamples(i, data[i]) = 1;

    arma::uvec categories(numCategories);
    if (UseWeights)
    {
        // Weight counts for computing the gain.
        double totalWeight = accu(weights);
        arma::mat classWeightSums = arma::zeros<arma::mat>(numClasses, 2);
        bestFoundGain *= totalWeight;

        // Recursively check the gain for all partitions.
        improved = PartitionSplit(
            data, labels, numCategories, numClasses, weights,
            totalWeight, bestFoundGain, categorySamples, categories,
            splitInfo, classWeightSums);
        bestFoundGain /= totalWeight;
    }
    else
    {
        // Class counts for computing the gain.
        arma::Mat<size_t> classCounts
            = arma::zeros<arma::Mat<size_t>>(numClasses, 2);
        bestFoundGain *= n;

        // Recursively check the gain for all partitions.
        improved = PartitionSplit(
            data, labels, numCategories, numClasses, bestFoundGain,
            categorySamples, categories, splitInfo, classCounts);
        bestFoundGain /= n;
    }
  }
  return improved ? bestFoundGain : DBL_MAX;
}


// Overload used in regression with MSEGain.
template<typename FitnessFunction>
template<bool UseWeights, typename VecType, typename ResponsesType,
         typename WeightVecType>
double BestBinaryCategoricalSplit<FitnessFunction>::SplitIfBetter(
    const double bestGain,
    const VecType& data,
    const size_t numCategories,
    const ResponsesType& responses,
    const WeightVecType& weights,
    const size_t minLeafSize,
    const double minGainSplit,
    arma::vec& splitInfo,
    AuxiliarySplitInfo& aux,
    FitnessFunction& fitnessFunction)
{
  static_assert(std::is_same_v<FitnessFunction, MSEGain>,
      "BestBinaryCategoricalSplit: regression FitnessFunction must be "
      "MSEGain.");
  const size_t n = data.n_elem;
  double bestFoundGain = std::min(bestGain + minGainSplit, 0.0);

  // We are too small to split.
  if (n < (minLeafSize * 2))
    return DBL_MAX;

  // Order the categories of variable vₖ by increasing mean
  // of the response y. categoryResponse[i, 0] will contain
  // the mean response for category Cᵢ.
  arma::vec categoryResponse(numCategories);
  arma::uvec categoryCounts(numCategories);

  for (size_t i = 0; i < n; ++i)
  {
    categoryResponse[data[i]] += responses[i];
    ++categoryCounts[data[i]];
  }
  for (size_t i = 0; i < numCategories; ++i)
  {
    categoryResponse[i] = categoryCounts[i] == 0 ? 0 :
        categoryResponse[i] / categoryCounts[i];
  }

  arma::uvec sortedCategories = sort_index(categoryResponse);
  arma::uvec categoryRank(numCategories);
  for (size_t i = 0; i < numCategories; i++)
     categoryRank[sortedCategories[i]] = i;

  arma::uvec transformedData(n);
  for (size_t i = 0; i < n; ++i)
    transformedData[i] = categoryRank[data[i]];

  // Split the transformed vₖ as a numeric type.
  bestFoundGain = NumericSplit::template SplitIfBetter<UseWeights>(
      bestFoundGain,
      transformedData,
      responses,
      weights,
      minLeafSize,
      minGainSplit,
      splitInfo,
      (NumericAux &) aux,
      fitnessFunction);
  bool improved = bestFoundGain != DBL_MAX;

  if (improved)
  {
      // This split is better: store the set membership in splitInfo
      // and return. Thus splitInfo is a vector of size Q, where Q is
      // the number of categories, and splitInfo[k] is zero if category
      // k is assigned to the left child, and otherwise it is one if k
      // is assigned to the right.
      size_t splitIndex = (size_t) std::floor((double) splitInfo[0]);
      splitInfo.set_size(numCategories + 1);
      for (size_t c : sortedCategories.subvec(0, splitIndex))
        splitInfo[c] = 0;
      for (size_t c : sortedCategories.subvec(splitIndex + 1,
                                              numCategories - 1))
        splitInfo[c] = 1;
      return bestFoundGain;
  }
  return DBL_MAX;
}

template<typename FitnessFunction>
template<typename VecType, typename LabelsType>
bool BestBinaryCategoricalSplit<FitnessFunction>::PartitionSplit(
    const VecType& data,
    const LabelsType& labels,
    const size_t numCategories,
    const size_t numClasses,
    double& bestFoundGain,
    arma::SpMat<short>& categorySamples,
    arma::uvec& categories,
    arma::vec& splitInfo,
    arma::Mat<size_t>& classCounts,
    size_t totalLeft,
    size_t totalRight,
    size_t k)
{
  /* Base case -- We have already found an optimal split. */
  if (bestFoundGain >= 0.0)
    return false;

  if (k == numCategories)
  {
    /**
     * Base case -- Compute the gain for the current partition.
     */
    double leftGain, rightGain, gain;

    // The gain for children Tₗ and Tᵣ.
    leftGain = FitnessFunction::template EvaluatePtr<false>(
        classCounts.colptr(0), numClasses, totalLeft);
    rightGain = FitnessFunction::template EvaluatePtr<false>(
        classCounts.colptr(1), numClasses, totalRight);
    // The gain for this split.
    gain = double(totalLeft) * leftGain + double(totalRight) * rightGain;

    // This is the best split found thus far.
    if (gain > bestFoundGain)
    {
      bestFoundGain = gain;
      splitInfo.set_size(numCategories);
      for (size_t i = 0; i < numCategories; ++i)
        splitInfo[i] = categories[i];
      return true;
    }
    return false;
  }
  bool improved = false;
  arma::uvec samples = find(categorySamples.col(k));

  /**
   * Compute the gain with category Cₖ ∈ Tₗ
   */
  categories[k] = 0;
  for (arma::uword i : samples)
  {
    ++classCounts(labels[i], 0);
  }
  totalLeft += samples.n_elem;

  improved = PartitionSplit(
      data, labels, numCategories, numClasses, bestFoundGain, categorySamples,
      categories, splitInfo, classCounts, totalLeft, totalRight, k + 1);

  /**
   * Compute the gain with category Cₖ ∈ Tᵣ
   */
  categories[k] = 1;
  for (arma::uword i : samples)
  {
    --classCounts(labels[i], 0);
    ++classCounts(labels[i], 1);
  }
  totalLeft -= samples.n_elem;
  totalRight += samples.n_elem;

  improved |= PartitionSplit(
      data, labels, numCategories, numClasses, bestFoundGain, categorySamples,
      categories, splitInfo, classCounts, totalLeft, totalRight, k + 1);

  /* Unassign Cₖ and return whether a better split was found. */
  for (arma::uword i : samples)
  {
    --classCounts(labels[i], 1);
  }
  totalRight -= samples.n_elem;
  return improved;
}

// Overload used with weights.
template<typename FitnessFunction>
template<typename VecType, typename LabelsType, typename WeightVecType>
bool BestBinaryCategoricalSplit<FitnessFunction>::PartitionSplit(
      const VecType& data,
      const LabelsType& labels,
      const size_t numCategories,
      const size_t numClasses,
      const WeightVecType& weights,
      const double totalWeight,
      double& bestFoundGain,
      arma::SpMat<short>& categorySamples,
      arma::uvec& categories,
      arma::vec& splitInfo,
      arma::mat& classWeightSums,
      double totalLeftWeight,
      double totalRightWeight,
      size_t k)
{
  /* Base case -- We have already found an optimal split. */
  if (bestFoundGain >= 0.0)
    return false;

  /**
   * Base case -- Compute the gain for the current partition.
   */
  if (k == numCategories)
  {
    double leftGain, rightGain, gain;

    // The gain for children Tₗ and Tᵣ.
    leftGain = FitnessFunction::template EvaluatePtr<true>(
        classWeightSums.colptr(0), numClasses, totalLeftWeight);
    rightGain = FitnessFunction::template EvaluatePtr<true>(
        classWeightSums.colptr(1), numClasses, totalRightWeight);

    // The gain for this split.
    gain = (totalLeftWeight * leftGain) + (totalRightWeight * rightGain);

    // This is the best split found thus far.
    if (gain > bestFoundGain)
    {
      bestFoundGain = gain;
      splitInfo.set_size(numCategories);
      for (size_t i = 0; i < numCategories; ++i)
        splitInfo[i] = categories[i];
      return true;
    }
    return false;
  }
  bool improved = false;
  arma::uvec samples = find(categorySamples.col(k));

  /**
   * Compute the gain with category Cₖ ∈ Tₗ
   */
  categories[k] = 0;
  for (arma::uword i : samples)
  {
    classWeightSums(labels[i], 0) += weights[i];
    totalLeftWeight += weights[i];
  }
  improved = PartitionSplit(
      data, labels, numCategories, numClasses, weights, totalWeight,
      bestFoundGain, categorySamples, categories, splitInfo,
      classWeightSums, totalLeftWeight, totalRightWeight, k + 1);

  /**
   * Compute the gain with category Cₖ ∈ Tᵣ
   */
  categories[k] = 1;
  for (arma::uword i : samples)
  {
    classWeightSums(labels[i], 0) -= weights[i];
    classWeightSums(labels[i], 1) += weights[i];
    totalLeftWeight -= weights[i];
    totalRightWeight += weights[i];
  }
  improved |= PartitionSplit(
      data, labels, numCategories, numClasses, weights, totalWeight,
      bestFoundGain, categorySamples, categories, splitInfo,
      classWeightSums, totalLeftWeight, totalRightWeight, k + 1);

  /* Unassign Cₖ and return whether a better split was found. */
  for (arma::uword i : samples)
  {
    classWeightSums(labels[i], 1) -= weights[i];
    totalRightWeight -= weights[i];
  }
  return improved;
}

} // namespace mlpack
#endif
