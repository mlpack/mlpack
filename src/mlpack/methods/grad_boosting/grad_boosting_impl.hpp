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
    numModels(0),
    numClasses(0)
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
               const size_t numModels,
               const size_t numClasses) :
  numModels(numModels),
  numClasses(numClasses)
{
  TrainInternal(data, labels, numModels, numClasses);
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
GradBoosting<MatType>::
GradBoosting(const MatType& data,
             const arma::Row<size_t>& labels,
             const size_t numModels,
             const size_t numClasses,
             const size_t minimumLeafSize, 
             const double minimumGainSplit, 
             const size_t maximumDepth) :
  numModels(numModels),
  numClasses(numClasses)
{
  TrainInternal(data, labels, numModels, numClasses, 
                minimumLeafSize, minimumGainSplit, maximumDepth);
}

template<typename MatType>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numModels,
        const size_t numClasses)
{
  return TrainInternal(data, labels, numModels, numClasses);
}

template<typename MatType>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numModels,
        const size_t numClasses,
        const size_t minimumLeafSize, 
        const double minimumGainSplit, 
        const size_t maximumDepth)
{
  return TrainInternal(data, labels, numModels, numClasses,
                       minimumLeafSize, minimumGainSplit, maximumDepth);
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


// TrainInternal is a private function within GradBoosting class
template<typename MatType>
void GradBoosting<MatType>::TrainInternal(const MatType& data,
                                          const arma::Row<size_t>& labels,
                                          const size_t numModels,
                                          const size_t numClasses,
                                          const size_t minimumLeafSize,
                                          const double minimumGainSplit,
                                          const size_t maximumDepth) 
{

  // Initiate dimensionSelector.
  const AllDimensionSelect dimensionSelector;

  // Initiate weights - not going to use.
  arma::mat weights;

  // Initiate learning rate.
  double learningRate = 0.1;

  // Clear the weak learners vector to in case it's preinitialised.
  weakLearners.clear();

  // Store residues.
  MatType residue(data.n_rows, numClasses, arma::fill::zeros);
  for (size_t i = 0; i < labels.n_cols; ++i) 
    residue(i, labels(i)) = 1;

  for (size_t model = 0; model < numModels; ++model) 
  {

    WeakLearnerType* wPtr;
    wPtr = new WeakLearnerType(data, labels, numClasses, 
    minimumLeafSize, minimumGainSplit, maximumDepth, dimensionSelector);

    WeakLearnerType w = *wPtr;
    delete wPtr;

    weakLearners.push_back(w);

    arma::Row<size_t> predictions(residue.n_cols, arma::fill::zeros);
    arma::mat probabilities;
    w.Classify(data, predictions, probabilities);

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

