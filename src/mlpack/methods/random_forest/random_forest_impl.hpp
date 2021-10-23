/**
 * @file methods/random_forest/random_forest_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of random forest.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_IMPL_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_IMPL_HPP

// In case it hasn't been included yet.
#include "random_forest.hpp"

namespace mlpack {
namespace tree {

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::RandomForest() :
    avgGain(0.0)
{
  // Nothing to do here.
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::RandomForest(const MatType& dataset,
                const LabelsType& labels,
                const size_t numClasses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, false>(dataset, info, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector,
      false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::RandomForest(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const LabelsType& labels,
                const size_t numClasses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector):
                    avgGain(0.0)
{
  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::RandomForest(const MatType& dataset,
                const LabelsType& labels,
                const size_t numClasses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored by Train().
  Train<true, false>(dataset, info, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector,
      false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::RandomForest(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const LabelsType& labels,
                const size_t numClasses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  Train<true, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::Train(const MatType& dataset,
         const LabelsType& labels,
         const size_t numClasses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  data::DatasetInfo datasetInfo; // Ignored by Train().
  arma::rowvec weights; // Ignored by Train().
  return Train<false, false>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const LabelsType& labels,
         const size_t numClasses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  arma::rowvec weights; // Ignored by Train().
  return Train<false, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::Train(const MatType& dataset,
         const LabelsType& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  data::DatasetInfo datasetInfo; // Ignored by Train().
  return Train<false, false>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename MatType, typename LabelsType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const LabelsType& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  return Train<true, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<typename Archive>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::serialize(Archive& ar, const uint32_t /* version */)
{
  size_t numTrees;
  if (cereal::is_loading<Archive>())
    trees.clear();
  else
    numTrees = trees.size();

  ar(CEREAL_NVP(numTrees));

  // Allocate space if needed.
  if (cereal::is_loading<Archive>())
    trees.resize(numTrees);

  ar(CEREAL_NVP(trees));
  ar(CEREAL_NVP(avgGain));
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename DecisionTreeType
>
template<bool UseWeights, bool UseDatasetInfo, typename MatType,
    typename LabelsType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    DecisionTreeType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const LabelsType& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType& dimensionSelector,
         const bool warmStart)
{
  // Reset the forest if we are not doing a warm-start.
  if (!warmStart)
    trees.clear();
  const size_t oldNumTrees = trees.size();
  trees.resize(trees.size() + numTrees);

  // Convert avgGain to total gain.
  double totalGain = avgGain * oldNumTrees;

  // Train each tree individually.
  #pragma omp parallel for reduction( + : totalGain)
  for (omp_size_t i = 0; i < numTrees; ++i)
  {
    MatType bootstrapDataset;
    LabelsType bootstrapLabels;
    arma::rowvec bootstrapWeights;
    if (UseBootstrap)
    {
      Bootstrap<UseWeights>(dataset, labels, weights, bootstrapDataset,
          bootstrapLabels, bootstrapWeights);
    }

    if (UseWeights)
    {
      if (UseDatasetInfo)
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, datasetInfo,
                bootstrapLabels, numClasses, bootstrapWeights, minimumLeafSize,
                minimumGainSplit, maximumDepth, dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, datasetInfo, labels,
                numClasses, weights, minimumLeafSize, minimumGainSplit,
                maximumDepth, dimensionSelector);
      }
      else
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, bootstrapLabels,
                numClasses, bootstrapWeights, minimumLeafSize,
                minimumGainSplit, maximumDepth, dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, labels, numClasses,
                weights, minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector);
      }
    }
    else
    {
      if (UseDatasetInfo)
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, datasetInfo,
                bootstrapLabels, numClasses, minimumLeafSize, minimumGainSplit,
                maximumDepth, dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, datasetInfo, labels,
                numClasses, minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector);
      }
      else
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, bootstrapLabels,
                numClasses, minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, labels, numClasses,
                minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector);
      }
    }
  }

  avgGain = totalGain / trees.size();
  return avgGain;
}

} // namespace tree
} // namespace mlpack

#endif
