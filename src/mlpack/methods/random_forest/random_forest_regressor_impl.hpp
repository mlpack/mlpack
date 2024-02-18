/**
 * @file methods/random_forest/random_forest_regressor_impl.hpp
 * @author Dinesh Kumar
 *
 * Implementation of random forest regressor.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_REGRESSOR_IMPL_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_REGRESSOR_IMPL_HPP

// In case it hasn't been included yet.
#include "random_forest_regressor.hpp"

namespace mlpack {

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::RandomForestRegressor() :
    avgGain(0.0)
{
  // Nothing to do here.
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::RandomForestRegressor(const MatType& dataset,
                const arma::Row<double>& responses,
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
  Train<false, false>(dataset, info, responses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector,
      false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::RandomForestRegressor(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const arma::Row<double>& responses,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector):
                    avgGain(0.0)
{
  // Pass off work to the Train() method.
  arma::rowvec weights; // Fake weights, not used.
  Train<false, true>(dataset, datasetInfo, responses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::RandomForestRegressor(const MatType& dataset,
                const arma::Row<double>& responses,
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
  Train<true, false>(dataset, info, responses, weights, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector,
      false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::RandomForestRegressor(const MatType& dataset,
                const data::DatasetInfo& datasetInfo,
                const arma::Row<double>& responses,
                const arma::rowvec& weights,
                const size_t numTrees,
                const size_t minimumLeafSize,
                const double minimumGainSplit,
                const size_t maximumDepth,
                DimensionSelectionType dimensionSelector) :
    avgGain(0.0)
{
  // Pass off work to the Train() method.
  Train<true, true>(dataset, datasetInfo, responses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, false);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
double RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Train(const MatType& dataset,
         const arma::Row<double>& responses,
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
  return Train<false, false>(dataset, datasetInfo, responses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
double RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<double>& responses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  arma::rowvec weights; // Ignored by Train().
  return Train<false, true>(dataset, datasetInfo, responses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
double RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Train(const MatType& dataset,
         const arma::Row<double>& responses,
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
  return Train<false, false>(dataset, datasetInfo, responses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
double RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<double>& responses,
         const arma::rowvec& weights,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         const bool warmStart,
         DimensionSelectionType dimensionSelector)
{
  // Pass off to Train().
  return Train<true, true>(dataset, datasetInfo, responses, weights,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector, warmStart);
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename VecType>
size_t RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Predict(const VecType& point) const
{
   // Check edge case.
  if (trees.size() == 0)
  { 
    throw std::invalid_argument("RandomForestRegressor::Predict(): no random forest "
        "trained!");
    return 0;
  }

  size_t prediction;
  size_t totalSum = 0;
  for (size_t i = 0; i < trees.size(); ++i)
  {
    totalSum += trees[i].Predict(point);
  }

  // Find maximum element after renormalizing probabilities.
  prediction = totalSum / trees.size();

  return prediction;
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename MatType>
void RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Predict(const MatType& data,
            arma::Row<double>& predictions) const
{
  // Check edge case.
  if (trees.size() == 0)
  {
    predictions.clear();

    throw std::invalid_argument("RandomForestRegressor::Predict(): no random forest "
        "trained!");
  }

  predictions.set_size(data.n_cols);

  #pragma omp parallel for
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    predictions[i] = Predict(data.col(i));
  }
}

template<
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType,
    bool UseBootstrap
>
template<typename Archive>
void RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
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
    bool UseBootstrap
>
template<bool UseWeights, bool UseDatasetInfo, typename MatType>
double RandomForestRegressor<
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType,
    UseBootstrap
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<double>& responses,
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
    arma::Row<double> bootstrapresponses;
    arma::rowvec bootstrapWeights;
    if (UseBootstrap)
    {
      Bootstrap<UseWeights>(dataset, responses, weights, bootstrapDataset,
          bootstrapresponses, bootstrapWeights);
    }

    if (UseWeights)
    {
      if (UseDatasetInfo)
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, datasetInfo,
                bootstrapresponses, bootstrapWeights, minimumLeafSize,
                minimumGainSplit, maximumDepth, dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, datasetInfo, responses,
                weights, minimumLeafSize, minimumGainSplit,
                maximumDepth, dimensionSelector);
      }
      else
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, bootstrapresponses,
                bootstrapWeights, minimumLeafSize,
                minimumGainSplit, maximumDepth, dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, responses,
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
                bootstrapresponses, minimumLeafSize, minimumGainSplit,
                maximumDepth, dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, datasetInfo, responses,
                minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector);
      }
      else
      {
        totalGain += UseBootstrap ?
            trees[oldNumTrees + i].Train(bootstrapDataset, bootstrapresponses,
                minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector) :
            trees[oldNumTrees + i].Train(dataset, responses,
                minimumLeafSize, minimumGainSplit, maximumDepth,
                dimensionSelector);
      }
    }
  }

  avgGain = totalGain / trees.size();
  return avgGain;
}

} // namespace mlpack

#endif
