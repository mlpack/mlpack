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
    template<typename> class CategoricalSplitType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::RandomForest(const MatType& dataset,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, false>(dataset, info, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::RandomForest(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::RandomForest(const MatType& dataset,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored by Train().
  Train<true, false>(dataset, info, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::RandomForest(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method.
  Train<true, true>(dataset, datasetInfo, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  data::DatasetInfo info; // Ignored by Train().
  arma::rowvec weights; // Ignored by Train().
  return Train<false, false>(dataset, info, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  arma::rowvec weights; // Ignored by Train().
  return Train<false, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  data::DatasetInfo info; // Ignored by Train().
  return Train<false, false>(dataset, info, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  return Train<true, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
size_t RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point) const
{
  // Pass off to another Classify() overload.
  size_t predictedClass;
  arma::vec probabilities;
  Classify(point, predictedClass, probabilities);

  return predictedClass;
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename VecType>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const VecType& point,
            size_t& prediction,
            arma::vec& probabilities) const
{
  // Check edge case.
  if (trees.size() == 0)
  {
    probabilities.clear();
    prediction = 0;

    throw std::invalid_argument("RandomForest::Classify(): no random forest "
        "trained!");
  }

  probabilities.zeros(trees[0].NumClasses());
  for (size_t i = 0; i < trees.size(); ++i)
  {
    arma::vec treeProbs;
    size_t treePrediction; // Ignored.
    trees[i].Classify(point, treePrediction, treeProbs);

    probabilities += treeProbs;
  }

  // Find maximum element after renormalizing probabilities.
  probabilities /= trees.size();
  arma::uword maxIndex = 0;
  probabilities.max(maxIndex);

  // Set prediction.
  prediction = (size_t) maxIndex;
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const MatType& data,
            arma::Row<size_t>& predictions) const
{
  // Check edge case.
  if (trees.size() == 0)
  {
    predictions.clear();

    throw std::invalid_argument("RandomForest::Classify(): no random forest "
        "trained!");
  }

  predictions.set_size(data.n_cols);

  #pragma omp parallel for
  for (omp_size_t i = 0; i < data.n_cols; ++i)
  {
    predictions[i] = Classify(data.col(i));
  }
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Classify(const MatType& data,
            arma::Row<size_t>& predictions,
            arma::mat& probabilities) const
{
  // Check edge case.
  if (trees.size() == 0)
  {
    predictions.clear();
    probabilities.clear();

    throw std::invalid_argument("RandomForest::Classify(): no random forest "
        "trained!");
  }

  probabilities.set_size(trees[0].NumClasses(), data.n_cols);
  predictions.set_size(data.n_cols);
  #pragma omp parallel for
  for (omp_size_t i = 0; i < data.n_cols; ++i)
  {
    arma::vec probs = probabilities.unsafe_col(i);
    Classify(data.col(i), predictions[i], probs);
  }
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Archive>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
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
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<bool UseWeights, bool UseDatasetInfo, typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType& dimensionSelector)
{
  // Train each tree individually.
  trees.resize(numTrees); // This will fill the vector with untrained trees.
  double avgGain = 0.0;

  #pragma omp parallel for reduction( + : avgGain)
  for (omp_size_t i = 0; i < numTrees; ++i)
  {
    Timer::Start("bootstrap");
    MatType bootstrapDataset;
    arma::Row<size_t> bootstrapLabels;
    arma::rowvec bootstrapWeights;
    Bootstrap<UseWeights>(dataset, labels, weights, bootstrapDataset,
        bootstrapLabels, bootstrapWeights);
    Timer::Stop("bootstrap");

    // Now build the decision tree.
    Timer::Start("train_tree");
    if (UseWeights)
    {
      if (UseDatasetInfo)
      {
        avgGain += trees[i].Train(bootstrapDataset, datasetInfo,
            bootstrapLabels, numClasses, bootstrapWeights, minimumLeafSize,
            minimumGainSplit, maximumDepth, dimensionSelector);
      }
      else
      {
        avgGain += trees[i].Train(bootstrapDataset, bootstrapLabels, numClasses,
            bootstrapWeights, minimumLeafSize, minimumGainSplit, maximumDepth,
            dimensionSelector);
      }
    }
    else
    {
      if (UseDatasetInfo)
      {
        avgGain += trees[i].Train(bootstrapDataset, datasetInfo,
            bootstrapLabels, numClasses, minimumLeafSize, minimumGainSplit,
            maximumDepth, dimensionSelector);
      }
      else
      {
        avgGain += trees[i].Train(bootstrapDataset, bootstrapLabels, numClasses,
            minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
      }
    }
    Timer::Stop("train_tree");
  }
  return avgGain / numTrees;
}

} // namespace tree
} // namespace mlpack

#endif
