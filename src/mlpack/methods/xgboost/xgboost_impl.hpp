/*
 * @file methods/xgboost/xgboost_impl.hpp
 * @author Abhimanyu Dayal
 *
 * Implementation of the XGBoost class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_XGBOOST_XGBOOST_IMPL_HPP
#define MLPACK_METHODS_XGBOOST_XGBOOST_IMPL_HPP

// Base definition of the XGBoostModel class.
#include "xgboost.hpp"
#include "loss_function.hpp"
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

// Empty constructor.
template<typename MatType>
XGBoost<MatType>::XGBoost() :
    numClasses(0),
    numModels(0)
{ /*Nothing to do.*/}

// In case trees aren't defined
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numClasses Number of classes
 * @param numModels Number of trees
 */
template<typename MatType>
XGBoost<MatType>::
  XGBoost(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numModels) :
  numClasses(numClasses),
  numModels(numModels)
{
  adjustments.resize(numModels);
  TrainInternal(data, labels, numClasses, numModels);
}

// In case the user inputs the arguments for the Weak Learner
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numClasses Number of classes
 * @param numModels Number of trees
 * @param other Weak Learner, which has been initialized already.
 */
template<typename MatType>
XGBoost<MatType>::
XGBoost(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numModels,
             const size_t minimumLeafSize, 
             const double minimumGainSplit, 
             const size_t maximumDepth) :
  numClasses(numClasses),
  numModels(numModels)
{
  adjustments.resize(numModels);
  TrainInternal(data, labels, numClasses, numModels, 
                minimumLeafSize, minimumGainSplit, maximumDepth);
}

template<typename MatType>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels)
{
  adjustments.resize(numModels);
  SetNumModels(numModels);
  return TrainInternal(data, labels, numClasses, numModels);
}

template<typename MatType>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels,
        const size_t minimumLeafSize, 
        const double minimumGainSplit, 
        const size_t maximumDepth)
{
  adjustments.resize(numModels);
  SetNumModels(numModels);
  return TrainInternal(data, labels, numClasses, numModels,
                       minimumLeafSize, minimumGainSplit, maximumDepth);
}

// Classify the given test point.
template<typename MatType>
template<typename VecType>
size_t XGBoost<MatType>::Classify(const VecType& point) 
{
  size_t prediction;
  Classify<arma::colvec>(point, prediction);

  return prediction;
}

template<typename MatType>
template<typename VecType>
void XGBoost<MatType>::Classify(const VecType& point,
                                      size_t& prediction)
{
  int tempPrediction = 0;

  for (size_t i = 0; i < trees.size(); ++i) 
  {
    size_t p;

    arma::vec tempProb(numClasses, arma::fill::zeros); /*Will not use*/
    trees[i].Classify(point, p, tempProb);
    tempPrediction -= adjustments(i);
    tempPrediction += p;
  }

  prediction = (size_t) tempPrediction;
}

template<typename MatType>
void XGBoost<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels) 
{
  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);
  
  for (size_t i = 0; i < test.n_cols; ++i) 
  {
    arma::vec tempData(test.n_rows); 
    size_t prediction;
    arma::vec tempProb(numClasses, arma::fill::zeros);

    for (size_t j = 0; j < test.n_rows; ++j)
      tempData(j) = test(j, i);

    Classify<arma::vec>(tempData, prediction);
    predictedLabels(i) = prediction;
  }

}


// TrainInternal is a private function within XGBoost class
template<typename MatType>
void XGBoost<MatType>::TrainInternal(const MatType& data,
                                          const arma::Row<size_t>& labels,
                                          const size_t numClasses,
                                          const size_t numModels,
                                          const size_t minimumLeafSize,
                                          const double minimumGainSplit,
                                          const size_t maximumDepth) 
{

  // Initiate dimensionSelector.
  const AllDimensionSelect dimensionSelector;

  // Clear the trees vector to in case it's preinitialised.
  trees.clear();

  // Store residues.
  arma::Row<size_t> residue = labels;

  for (size_t model = 0; model < numModels; ++model) 
  {

    arma::Row<size_t> tempResidue = arma::unique(residue);
    size_t tempNumClasses = tempResidue.n_elem;

    XGBTree w(data, residue, tempNumClasses, 
    minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);

    trees.push_back(w);

    arma::Row<size_t> predictions(residue.n_elem, arma::fill::zeros);
    arma::mat probabilities; /*Will not use*/
    w.Classify(data, predictions, probabilities);

    if (model != numModels - 1)
    {
      int minDifference = 1e9;
      for (size_t i = 0; i < residue.n_elem; ++i)
      {
        int tempDifference = (int) residue(i) - (int) predictions(i);
        minDifference = std::min(tempDifference, minDifference);
      }

      adjustments(model) = -minDifference;
      residue = residue + adjustments(model);

      for (size_t i = 0; i < residue.n_elem; ++i) 
      {
        residue(i) = residue(i) - predictions(i);
      }
    }

  }
}

}

#endif