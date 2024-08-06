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
{
// Nothing to do.
}

// In case the user has already initialised the weak learner
// Weak learner type "WeakLearnerInType" defined by the template
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numModels Number of weak learners
 * @param numClasses Number of classes
 * @param other Weak Learner, which has been initialized already.
 */
template<typename MatType>
XGBoost<MatType>::
  XGBoost(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numModels,
                const XGBTree& other) :

  numClasses(numClasses),
  numModels(numModels)

{
  (void) TrainInternal<true>(data, labels, numClasses, other);
}

// In case the user inputs the arguments for the Weak Learner
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numModels Number of weak learners
 * @param numClasses Number of classes
 * @param other Weak Learner, which has been initialized already.
 */
template<typename MatType>

// Variadic template to the Weak Learner arguments
template<typename... WeakLearnerArgs>
XGBoost<MatType>::
XGBoost(const MatType& data,
              const arma::Row<size_t>& labels,
              const size_t numClasses,
              const size_t numModels,
              WeakLearnerArgs&&... weakLearnerArgs) :
  numModels(numModels)
{
  XGBTree other; // Will not be used.
  (void) TrainInternal<false>(data, labels, numClasses, other,
      weakLearnerArgs...);
}

// Train XGBoost with a given weak learner.

template<typename MatType>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels,
        const XGBTree learner)
{
  return TrainInternal<true>(data, labels, numModels, numClasses, learner);
}

template<typename MatType>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels)
{
  XGBTree other; // Will not be used.
  return TrainInternal<false>(data, labels, numModels, numClasses, other);
}

template<typename MatType>
template<typename... WeakLearnerArgs>
void XGBoost<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels,
        WeakLearnerArgs&&... weakLearnerArgs)
{
  XGBTree other; // Will not be used.
  return TrainInternal<false>(data, labels, numModels, numClasses, other,
    weakLearnerArgs...);
}

// Classify the given test point.
template<typename MatType>
template<typename VecType>
size_t XGBoost<MatType>::Classify(const VecType& point) 
{
  size_t prediction;
  arma::Row<ElemType> probabilities;
  Classify(point, prediction, probabilities);
  return prediction;
}

template<typename MatType>
template<typename VecType>
void XGBoost<MatType>::Classify(const VecType& point,
                                                      size_t& prediction,
                                                      arma::Row<ElemType>& probabilities)
{

  prediction = 0;

  for (size_t i = 0; i < weakLearners.size(); ++i) 
  {
    size_t tempPred = weakLearners[i].Classify(point);
    prediction += tempPred;
  }

}


template<typename MatType>
void XGBoost<MatType>::Classify(const MatType& test,
                                                      arma::Row<size_t>& predictedLabels) 
{

  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);
  arma::Row<ElemType> probabilities;

  for (size_t i = 0; i < test.n_cols; ++i) 
  {
    size_t prediction;
    Classify<arma::colvec>(test.col(i), prediction, probabilities);
    predictedLabels(i) = prediction;
  }
}


template<typename MatType>
void XGBoost<MatType>::Classify(const MatType& test,
                                                      arma::Row<size_t>& predictedLabels,
                                                      arma::Row<ElemType>& probabilities) 
{

  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);

  for (size_t i = 0; i < test.n_cols; ++i) 
  {
    size_t prediction;
    Classify<arma::colvec>(test.col(i), prediction, probabilities);
    predictedLabels(i) = prediction;
  }
}

template<typename MatType>
template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>
void XGBoost<MatType>:: 
  TrainInternal(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numModels,
                const size_t numClasses,
                const XGBTree& wl,
                WeakLearnerArgs&&... weakLearnerArgs) 
{

  // Load the initial weights into a 2-D matrix.
  const ElemType initWeight = 1.0 / ElemType(data.n_cols * numClasses);
  MatType D(numClasses, data.n_cols);
  D.fill(initWeight);

  // Weights are stored in this row vector.
  arma::Row<ElemType> weights(labels.n_cols);

  weakLearners.clear();

  MatType residue(data.n_rows, numClasses, arma::fill::zeros);

  for (size_t i = 0; i < labels.n_cols; ++i) 
  {
    residue(i, labels(i)) = 1;
  }

  for (size_t model = 0; model < numModels; ++model) 
  {

    // Build the weight vectors.
    weights = sum(D);

    XGBTree* wPtr;

    if(UseExistingWeakLearner)
    {
      wPtr = new XGBTree(wl, data, residue, numClasses, weights,
        std::forward<WeakLearnerArgs>(weakLearnerArgs)...);
    }
    else 
    {
      wPtr = new XGBTree(data, residue, numClasses,
        std::forward<WeakLearnerArgs>(weakLearnerArgs)...);
    }

    XGBTree w = *wPtr;

    arma::Row<size_t> predictions(residue.n_cols, arma::fill::zeros);
    w.Classify(data, predictions);
    w.Prune(pruningThreshold);

    weakLearners.push_back(w);

    // Compute the learning rate for this iteration.
    double learningRate = 0.1;

    for (size_t i = 0; i < residues.n_rows; ++i) 
    {
      for (size_t j = 0; j < residues.n_cols; ++j)
      {
        residue(i, j) = (size_t)abs((double)residue(i, j) - (double)(learningRate * probabilities(i, j)));
      }
    }
    delete wPtr;
        
  }
}

}

#endif