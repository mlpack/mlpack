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
#ifndef MLPACK_METHODS_GRAD_BOOSTING_GRADBOOSTING_IMPL_HPP
#define MLPACK_METHODS_GRAD_BOOSTING_GRADBOOSTING_IMPL_HPP

// Base definition of the GradBoostingModel class.
#include "grad_boosting.hpp"
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

// Empty constructor.
template<typename MatType>
GradBoosting<MatType>::GradBoosting() :
    numClasses(0),
    numWeakLearners(0)
{ /*Nothing to do.*/}

// In case weak learners aren't defined
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numClasses Number of classes
 * @param numWeakLearners Number of weak learners
 */
template<typename MatType>
GradBoosting<MatType>::
  GradBoosting(const MatType& data,
               const data::DatasetInfo& datasetInfo,
               const arma::Row<size_t>& labels,
               const size_t numClasses,
               const size_t numWeakLearners) :
  numClasses(numClasses),
  numWeakLearners(numWeakLearners)
{
  TrainInternal(data, datasetInfo, labels, numWeakLearners);
}

// In case the user inputs the arguments for the Weak Learner
/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param numClasses Number of classes
 * @param numWeakLearners Number of weak learners
 * @param other Weak Learner, which has been initialized already.
 */
template<typename MatType>
GradBoosting<MatType>::
GradBoosting(const MatType& data,
             const data::DatasetInfo& datasetInfo,
             const arma::Row<size_t>& labels,
             const size_t numClasses,
             const size_t numWeakLearners,
             const size_t minimumLeafSize,
             const double minimumGainSplit,
             const size_t maximumDepth) :
  numClasses(numClasses),
  numWeakLearners(numWeakLearners)
{
  TrainInternal(data, datasetInfo, labels, numWeakLearners,
                minimumLeafSize, minimumGainSplit, maximumDepth);
}

template<typename MatType>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const data::DatasetInfo& datasetInfo,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numWeakLearners)
{
  SetNumClasses(numClasses);
  NumWeakLearners(numWeakLearners);
  return TrainInternal(data, datasetInfo, labels, numWeakLearners);
}

template<typename MatType>
void GradBoosting<MatType>::
  Train(const MatType& data,
        const data::DatasetInfo& datasetInfo,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numWeakLearners,
        const size_t minimumLeafSize,
        const double minimumGainSplit,
        const size_t maximumDepth)
{
  SetNumClasses(numClasses);
  NumWeakLearners(numWeakLearners);
  return TrainInternal(data, datasetInfo, labels, numWeakLearners,
                       minimumLeafSize, minimumGainSplit, maximumDepth);
}

// Classify the given test point.
template<typename MatType>
template<typename VecType>
size_t GradBoosting<MatType>::Classify(const VecType& point)
{
  size_t prediction;
  Classify<arma::colvec>(point, prediction);

  return prediction;
}

template<typename MatType>
template<typename VecType>
void GradBoosting<MatType>::Classify(const VecType& point,
                                     size_t& prediction)
{
  double tempPrediction = 0;

  #pragma omp parallel for reduction(+:tempPrediction)
  for (size_t i = 0; i < weakLearners.size(); ++i)
  {
    double p = weakLearners[i]->Predict(point);
    tempPrediction += p;
  }

  for (double k = 0; k < numClasses; ++k)
  {
    if (std::abs(tempPrediction - k) < std::abs(tempPrediction - prediction))
      prediction = (size_t) k;
  }
}

template<typename MatType>
void GradBoosting<MatType>::Classify(const MatType& test,
                                     arma::Row<size_t>& predictedLabels)
{
  predictedLabels.clear();
  predictedLabels.resize(test.n_cols);

  for (size_t i = 0; i < test.n_cols; ++i)
  {
    size_t prediction;
    Classify(test.col(i), prediction);
    predictedLabels(i) = prediction;
  }
}


// TrainInternal is a private function within GradBoosting class
template<typename MatType>
void GradBoosting<MatType>::TrainInternal(const MatType& data,
                                          const data::DatasetInfo& datasetInfo,
                                          const arma::Row<size_t>& labels,
                                          const size_t numWeakLearners,
                                          const size_t minimumLeafSize,
                                          const double minimumGainSplit,
                                          const size_t maximumDepth)
{
  // Clear the weak learners vector to in case it's preinitialised.
  weakLearners.clear();
  // Store residues.
  arma::Row<double> residue(labels.n_cols);

  for (size_t i = 0; i < labels.n_cols; ++i)
    residue(i) = (double) labels(i);

  for (size_t model = 0; model < numWeakLearners; ++model)
  {
    WeakLearnerType* w = new WeakLearnerType(data,
                                             datasetInfo,
                                             residue,
                                             minimumLeafSize,
                                             minimumGainSplit,
                                             maximumDepth);

    weakLearners.push_back(w);

    arma::Row<double> predictions(residue.n_elem);
    w->Predict(data, predictions);

    if (model != numWeakLearners - 1)
      residue = residue - predictions;
  }
}

} // namespace mlpack

#endif
