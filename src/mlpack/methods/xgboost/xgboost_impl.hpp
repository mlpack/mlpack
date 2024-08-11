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
  arma::vec probabilities; /* Not to be used */
  Classify<arma::colvec>(point, prediction, probabilities);

  return prediction;
}

template<typename MatType>
template<typename VecType>
void XGBoost<MatType>::Classify(const VecType& point,
                                size_t& prediction,
                                VecType& probabilities)
{
  arma::rowvec rawScores(numClasses, arma::fill::zeros);

  for (size_t i = 0; i < trees.size(); ++i) 
  {
    arma::rowvec tempRawScores(numClasses, arma::fill::zeros);
    tree[i].Classify(point, tempRawScores);
    rawScores = rawScores + tempRawScores;
  }
  CalculateProbability(rawScores, probabilities);

  prediction = 0;
  for (size_t i = 0; i < numClasses; ++i)
  {
    if (probabilities(prediction) < probabilities(i))
      prediction = i;
  }
}

template<typename MatType>
void XGBoost<MatType>::Classify(const MatType& test,
                                arma::Row<size_t>& predictedLabels,
                                MatType& probabilities) 
{
  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);
  
  for (size_t i = 0; i < test.n_cols; ++i) 
  {
    size_t prediction;
    arma::vec tempData(test.n_rows); 
    arma::vec tempProb(numClasses, arma::fill::zeros);

    for (size_t j = 0; j < test.n_rows; ++j)
      tempData(j) = test(j, i);

    Classify<arma::vec>(tempData, prediction, tempProb);
    predictedLabels(i) = prediction;

    for (size_t j = 0; j < numClasses; ++j)
      probabilities(i, j) = tempProb(j);
  }

}


void CalculateProbability(arma::mat& rawScores,
                          arma::mat& probabilities)
{
  for (size_t i = 0; i < rawScores.n_rows; ++i)
  {
    double sumExp = 0;
    for (size_t k = 0; k < rawScores.n_cols; ++k)
      sumExp += std::exp(rawScores(i, k));
    
    for (size_t k = 0; k < rawScores.n_cols; ++k)  
      probabilities(i, k) = std::exp(rawScores(i, k)) / sumExp;
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

  // Initiate variables.
  double numClasses = (double) numClasses;
  double n = (double) labels.n_elems;
  double learningRate = 1;

  // Initiate matrices for use.
  arma::mat rawScores(numClasses, n, arma::fill::zeros);
  arma::mat probabilities(numClasses, n, arma::fill::value(1.0 / n));
  arma::mat residues(numClasses, n, arma::fill:zeros);
  arma::mat tempLabels(numClasses, n, arma::fill:zeros);

  // Preparing tempLabels.
  for (size_t i = 0; i < n; ++i)
    tempLabels(labels(i), i) = 1.0;

  for (size_t model = 0; model < numModels; ++model) 
  {
    // Residue is first calculated to train the weak learner.
    residues = tempLabels - probabilities;

    // The weak learner is trained.
    XGBTree w(data, residue, numClasses, 
    minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);

    // Raw scores are obtained to update probabilities variable.
    arma::mat tempPredictions(numClasses, n);
    w.Classify(data, tempPredictions);

    // Raw scores are updated.
    rawScores = rawScores + learningRate * tempPredictions;

    // Probabilities are calculated.
    CalculateProbability(rawScores, probabilities);

    // The weak learner is saved.
    trees.push_back(w);
  }
}

}

#endif