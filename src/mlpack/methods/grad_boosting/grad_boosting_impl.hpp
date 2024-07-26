/*
 * @file methods/grad_boosting/grad_boosting_impl.hpp
 * @author Abhimanyu Dayal
 *
 * Implementation of the Gradient Boosting class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Include guard - Used to prevent including double copies of the same code
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_IMPL_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_IMPL_HPP

// Base definition of the GradBoostingModel class.
#include "grad_boosting.hpp"
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

// Empty constructor.
template<typename MatType>
GradBoosting<MatType>::GradBoosting() :
    numClasses(0),
    numModels(0)
{ /*Nothing to do.*/}

// In case weak learners aren't defined
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numModels Number of weak learners
 * @param numClasses Number of classes
 */
template<typename MatType>
GradBoosting<MatType>::
  GradBoosting(const MatType& data,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numModels) :
  numClasses(numClasses),
  numModels(numModels)
{
  WeakLearnerType other; // Will not be used.
  (void) TrainInternal<false>(data, labels, numClasses, other);
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
GradBoosting<MatType>::
  GradBoosting(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numClasses,
                const size_t numModels,
                const WeakLearnerType& other) :
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
GradBoosting<MatType>::
GradBoosting(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numModels,
             WeakLearnerArgs&&... weakLearnerArgs) :
  numModels(numModels)
{
  WeakLearnerType other; // Will not be used.
  (void) TrainInternal<false>(data, labels, numClasses, other,
      weakLearnerArgs...);
}

// Train GradBoosting with a given weak learner.

template<typename MatType>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels,
        const WeakLearnerType learner)
{
  return TrainInternal<true>(data, labels, numModels, numClasses, learner);
}

template<typename MatType>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels)
{
  WeakLearnerType other; // Will not be used.
  return TrainInternal<false>(data, labels, numModels, numClasses, other);
}

template<typename MatType>
template<typename... WeakLearnerArgs>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels,
        WeakLearnerArgs&&... weakLearnerArgs)
{
  WeakLearnerType other; // Will not be used.
  return TrainInternal<false>(data, labels, numModels, numClasses, other,
    weakLearnerArgs...);
}

// Classify the given test point.
template<typename MatType>
template<typename VecType>
size_t GradBoosting<MatType>::Classify(const VecType& point) 
{
  size_t prediction;
  arma::Row<ElemType> probabilities;
  Classify<arma::colvec>(point, prediction, probabilities);

  return prediction;
}

template<typename MatType>
template<typename VecType>
void GradBoosting<MatType>::Classify(const VecType& point,
                                      size_t& prediction,
                                      arma::vec& probabilities)
{
  probabilities.resize(numClasses);

  for (size_t i = 0; i < weakLearners.size(); ++i) 
  {
    size_t p; /*Will not be used*/

    arma::vec tempProb(numClasses, arma::fill::zeros);
    weakLearners[i].Classify(point, p, tempProb);
    probabilities = probabilities + tempProb;
  }

  // Go through all the probabilities and return the 
  // variable with highest probability
  prediction = 0;
  for (size_t i = 0; i < probabilities.n_cols; i++)
  {
    if (probabilities(i) > probabilities(prediction))
      prediction = i;
  }

}

template<typename MatType>
void GradBoosting<MatType>::Classify(const MatType& test,
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
void GradBoosting<MatType>::Classify(const MatType& test,
                                      arma::Row<size_t>& predictedLabels,
                                      arma::Row<ElemType>& probabilities) const 
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

// This function uses default arguments
template<typename MatType>
template<bool UseExistingWeakLearner>
void GradBoosting<MatType>::TrainInternal(const MatType& data,
                                          const arma::Row<size_t>& labels,
                                          const size_t numModels,
                                          const size_t numClasses,
                                          const WeakLearnerType& wl)
{
  const size_t minimumLeafSize=10;
  const double minimumGainSplit=1e-7;
  const size_t maximumDepth=2;
  const AllDimensionSelect dimensionSelector;

  // Call the main TrainInternal function with default arguments
  return TrainInternal<UseExistingWeakLearner>(
      data, labels, numModels, numClasses, wl,
      minimumLeafSize,
      minimumGainSplit,
      maximumDepth,
      dimensionSelector);
}


// Template for GradBoosting template as a whole
template<typename MatType>

// Template for TrainInternal 
// UseExistingWeakLearner determines whether to define a weak learner anew or 
// use an existing weak learner
// WeakLearnerArgs are the arguments for the weak learner
template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>

// TrainInternal is a private function within GradBoosting class
// It has return type ElemType
void GradBoosting<MatType>:: 
  TrainInternal(const MatType& data,
                const arma::Row<size_t>& labels,
                const size_t numModels,
                const size_t numClasses,
                const WeakLearnerType& wl,
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

    WeakLearnerType* wPtr;

    if(UseExistingWeakLearner)
    {
      wPtr = new WeakLearnerType(wl, data, residue, numClasses, weights,
        std::forward<WeakLearnerArgs>(weakLearnerArgs)...);
    }
    else 
    {
      wPtr = new WeakLearnerType(data, residue, numClasses,
        std::forward<WeakLearnerArgs>(weakLearnerArgs)...);
    }

    WeakLearnerType w = *wPtr;
    delete wPtr;

    weakLearners.push_back(w);

    arma::Row<size_t> predictions(residue.n_cols, arma::fill::zeros);
    arma::mat probabilities;
    w.Classify(data, predictions, probabilities);

    // Compute the learning rate for this iteration.
    double learningRate = 0.1;

    for (size_t i = 0; i < residue.n_rows; ++i) 
    {
      for (size_t j = 0; j < residue.n_cols; ++j)
      {
        residue(i, j) = (size_t)abs((double)residue(i, j) - 
          (double)(learningRate * probabilities(i, j)));
      }
    }
  }
}

}

#endif

