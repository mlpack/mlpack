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

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
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
    typename BootstrapType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::RandomForest(const MatType& dataset,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector,
                BootstrapType bootstrap) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, false>(dataset, info, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, false,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::RandomForest(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector,
                BootstrapType bootstrap):
                    avgGain(0.0)
{
  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth, false,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::RandomForest(const MatType& dataset,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector,
                BootstrapType bootstrap) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored by Train().
  Train<true, false>(dataset, info, labels, numClasses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, false,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::RandomForest(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector,
                BootstrapType bootstrap) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  Train<true, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth, false,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::Train(const MatType& dataset,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector,
         BootstrapType bootstrap)
{
  // Pass off to Train().
  data::DatasetInfo datasetInfo; // Ignored by Train().
  arma::rowvec weights; // Ignored by Train().
  return
    Train<false, false>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth, warmStart,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector,
         BootstrapType bootstrap)
{
  // Pass off to Train().
  arma::rowvec weights; // Ignored by Train().
  return
    Train<false, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth, warmStart,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::Train(const MatType& dataset,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector,
         BootstrapType bootstrap)
{
  // Pass off to Train().
  data::DatasetInfo datasetInfo; // Ignored by Train().
  return
    Train<true, false>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth, warmStart,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector,
         BootstrapType bootstrap)
{
  // Pass off to Train().
  return
    Train<true, true>(dataset, datasetInfo, labels, numClasses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth, warmStart,
      dimensionSelector, bootstrap);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename VecType>
size_t RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
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
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename VecType>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
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
  arma::uword maxIndex = probabilities.index_max();

  // Set prediction.
  prediction = (size_t) maxIndex;
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
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
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    predictions[i] = Classify(data.col(i));
  }
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename MatType>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
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
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::vec probs = probabilities.unsafe_col(i);
    Classify(data.col(i), predictions[i], probs);
  }
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap,
    typename BootstrapType
>
template<typename Archive>
void RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
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
    typename BootstrapType
>
template<bool UseWeights, bool UseDatasetInfo, typename MatType>
double RandomForest<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap,
    BootstrapType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<size_t>& labels,
         const size_t numClasses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType& dimensionSelector,
         BootstrapType& bootstrap)
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
  for (size_t i = 0; i < numTrees; ++i)
  {
    // NOTE: this is a hacky workaround for older versions of Armadillo that did
    // not (by default) set a different seed for each RNG.  We simply manually
    // set the RNG seed for each individual thread.  However, if users have
    // specifically set the RNG seed for Armadillo, this could break their code.
    // So we hide it behind an (undocumented except for here) ifndef so that
    // this support can be disabled if desired...
    #undef MLPACK_ARMA_VERSION
    #define MLPACK_ARMA_VERSION (ARMA_VERSION_MAJOR * 100000 + \
                                 ARMA_VERSION_MINOR *    100 + \
                                 ARMA_VERSION_PATCH)
    #if (MLPACK_ARMA_VERSION < 1200602) // 12.6.2
      #ifndef MLPACK_DONT_OVERWRITE_ARMA_RNG_SEEDS
      // Note that each thread has its own differently-seeded RNG, so this will
      // result in a different seed for each thread's Armadillo RNG.
      arma::arma_rng::set_seed(RandGen()());
      #endif
    #endif

    MatType bootstrapDataset;
    arma::Row<size_t> bootstrapLabels;
    arma::rowvec bootstrapWeights;
    bootstrap.template Bootstrap<UseWeights>(dataset, labels, weights,
        bootstrapDataset, bootstrapLabels, bootstrapWeights);

    if (UseWeights)
    {
      if (UseDatasetInfo)
      {
        totalGain +=
            trees[oldNumTrees + i].Train(bootstrapDataset, datasetInfo,
                bootstrapLabels, numClasses, bootstrapWeights, minimumLeafSize,
                minimumGainSplit, maximumDepth, dimensionSelector);
      }
      else
      {
        totalGain +=
            trees[oldNumTrees + i].Train(bootstrapDataset, bootstrapLabels,
                numClasses, bootstrapWeights, minimumLeafSize,
                minimumGainSplit, maximumDepth, dimensionSelector);
      }
    }
    else
    {
      if (UseDatasetInfo)
      {
        totalGain +=
            trees[oldNumTrees + i].Train(bootstrapDataset, datasetInfo,
                bootstrapLabels, numClasses, minimumLeafSize, minimumGainSplit,
                maximumDepth, dimensionSelector);
      }
      else
      {
        totalGain +=
            trees[oldNumTrees + i].Train(bootstrapDataset, bootstrapLabels,
                numClasses, minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector);
      }
    }
  }

  avgGain = totalGain / trees.size();
  return avgGain;
}

} // namespace mlpack

#endif
