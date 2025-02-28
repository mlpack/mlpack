/**
 * @file methods/adaboost/adaboost_regressor_impl.hpp
 * @author Dinesh Kumar
 *
 * Implementation of Adaboost regressor class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ADABOOST_ADABOOST_REGRESSOR_IMPL_HPP
#define MLPACK_METHODS_ADABOOST_ADABOOST_REGRESSOR_IMPL_HPP

// In case it hasn't been included yet.
#include "adaboost_regressor.hpp"

namespace mlpack{

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::AdaBoostRegressor() : 
    avgGain(0.0)
{
  // Nothing to do here.
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType, typename ElemType>
AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::AdaBoostRegressor(const MatType& dataset,
                     const arma::Row<ElemType>& responses,
                     const size_t numTrees,
                     const size_t minimumLeafSize,
                     const double minimumGainSplit,
                     const size_t maximumDepth,
                     DimensionSelectionType dimensionSelector) :
                      avgGain(0.0)
{
  // Pass off work to the Train() method.
  data::DatasetInfo info; // Ignored.
  Train<false>(dataset, info, responses, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType, typename ElemType>
AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::AdaBoostRegressor(const MatType& dataset,
                     const data::DatasetInfo& datasetInfo,
                     const arma::Row<ElemType>& responses,
                     const size_t numTrees,
                     const size_t minimumLeafSize,
                     const double minimumGainSplit,
                     const size_t maximumDepth,
                     DimensionSelectionType dimensionSelector) :
                       avgGain(0.0)
{
  // Pass off work to the Train() method
  Train<true>(dataset, datasetInfo, responses,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType, typename ElemType>
double AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const arma::Row<ElemType>& responses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method
  data::DatasetInfo info; // Ignored.
  return Train<false>(dataset, info, responses, numTrees,
      minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType, typename ElemType>
double AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& dataset,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<ElemType>& responses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType dimensionSelector)
{
  // Pass off work to the Train() method
  return Train<true>(dataset, datasetInfo, responses,
      numTrees, minimumLeafSize, minimumGainSplit, maximumDepth,
      dimensionSelector);
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Vectype>
double AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Predict(const Vectype& point) const
{
   // Check edge case.
  if (trees.size() == 0)
  { 
    throw std::invalid_argument("AdaBoostRegressor::Predict(): no AdaBoost Regressor "
        "trained!");
    return 0;
  }

  arma::Row<double> predictions(trees.size());

  // Parallelly get predictions from all trees
  #pragma omp parallel for
  for (size_t i = 0; i < trees.size(); i++)
    predictions[i] = trees[i].Predict(point);

  predictions.replace(arma::datum::nan, 0);

  // Get sortedIndices of predictions.
  arma::uvec sortedIndices = arma::sort_index(predictions);
  arma::Row<double> sortedPred(predictions.size());
  arma::Row<double> sortedConf(confidence.size());

  // Match predictions and confidence according to sortedIndices.
  for (size_t i = 0; i < sortedIndices.size(); i++)
  {
    sortedPred[i] = predictions[sortedIndices[i]];
    sortedConf[i] = confidence[sortedIndices[i]];
  }

  // Calculate weighted-median of sortedConf.
  arma::Row<double> confCumSum = arma::cumsum(sortedConf);
  arma::uvec median = arma::find(confCumSum >= arma::accu(sortedConf)/2, 1);

  return predictions[median[0]]; 
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename MatType, typename ElemType>
void AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Predict(const MatType& data,
           arma::Row<ElemType>& predictions) const
{
  // Check edge case.
  if (trees.size() == 0)
  { 
    predictions.clear();

    throw std::invalid_argument("AdaBoostRegressor::Predict(): no AdaBoost Regressor "
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
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<typename Archive>
void AdaBoostRegressor<
    LossFunctionType,
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
  ar(CEREAL_NVP(avgGain));
}

template<
    typename LossFunctionType,
    typename FitnessFunction,
    typename DimensionSelectionType,
    template<typename> class NumericSplitType,
    template<typename> class CategoricalSplitType
>
template<bool UseDatasetInfo, typename MatType, typename ElemType>
double AdaBoostRegressor<
    LossFunctionType,
    FitnessFunction,
    DimensionSelectionType,
    NumericSplitType,
    CategoricalSplitType
>::Train(const MatType& data,
         const data::DatasetInfo& datasetInfo,
         const arma::Row<ElemType>& responses,
         const size_t numTrees,
         const size_t minimumLeafSize,
         const double minimumGainSplit,
         const size_t maximumDepth,
         DimensionSelectionType& dimensionSelector)
{
  // Clear trees.
  trees.clear();

  // Clear confidence of trees.
  confidence.clear();

  // Load initial weights to 1 / n_responses.
  arma::rowvec weights(responses.n_cols);
  weights.fill(1/(double)responses.n_cols);


  // To be used for prediction.
  arma::Row<ElemType> predictedValues(responses.n_cols);

  double totalGain = 0.0;

  for (size_t i=0; i < numTrees; ++i)
  {
    // avoid extremely small sample weight
    weights.clean(arma::datum::eps);

    DecisionTreeType tree;

    if (UseDatasetInfo)
    {
      totalGain += tree.Train(data, datasetInfo, responses, weights,
            minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);
    }
    else
    {
      totalGain += tree.Train(data, responses, weights, minimumLeafSize,
            minimumGainSplit, maximumDepth, dimensionSelector);
    }

    tree.Predict(data, predictedValues);

    // Calculate loss based on given lossFunction.
    arma::Row<ElemType> diff = predictedValues - responses;
    arma::Row<ElemType> errorVec = LossFunctionType::Calculate(diff);

    // Calculate the average loss.
    ElemType avgLoss = arma::accu(errorVec % weights);
    trees.push_back(tree);

    if (avgLoss <= 0)
    {
      confidence.push_back(1.0);
      break;
    }
    else if (avgLoss >= 0.5)
    {
      trees.pop_back();
      break;
    }
    
    double beta = avgLoss / (1.0 - avgLoss);

    // Boost weight using AdaBoost.R2 alg
    confidence.push_back(std::log(1.0 / beta));

    // Update weights.
    if (i != numTrees-1){
      #pragma omp parallel for
      for (size_t j = 0; j<errorVec.n_cols; j++)
        weights[j] *= std::pow(beta, 1.0 - errorVec[j]);
    }


    if(weights.has_inf()){
      Log::Warn << "Sample weights have reached infinite values,"
                << " at tree "<< i<<" , causing overflow. "
                << "Iterations stopped. Try lowering number of trees." << std::endl;
      break;
    }

    // Stop if the sum of weights has become non positive.
    if(arma::accu(weights)<=0)
      break;

    // Normalize weights
    weights = weights / arma::accu(weights);
  }

  return totalGain;
}

}

#endif
