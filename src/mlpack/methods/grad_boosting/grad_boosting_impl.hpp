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
#ifndef MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_IMPL_HPP
#define MLPACK_METHODS_GRADBOOSTING_GRADBOOSTING_IMPL_HPP

#include "grad_boosting.hpp"

namespace mlpack {

// Empty constructor.
template<typename WeakLearnerType, typename MatType>
GradBoosting<WeakLearnerType, MatType>::GradBoosting() :
    numClasses(0),
    num_models(0)
{
// Nothing to do.
}

/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param num_models Number of weak learners
 * @param other Weak Learner, which has been initialized already.
 */
template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
GradBoosting<MatType>::GradBoosting(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t num_models,
    const WeakLearnerInType& other,
    const typename std::enable_if<
        std::is_same<WeakLearnerType, WeakLearnerInType>::value>::type*) :
    num_models(num_models)
{
  (void) TrainInternal<true>(data, labels, numClasses, other);
}

/**
 * Constructor.
 *
 * @param data Input data
 * @param labels Corresponding labels
 * @param num_models Number of weak learners
 * @param other Weak Learner, which has been initialized already.
 */
template<typename WeakLearnerType, typename MatType>
template<typename... WeakLearnerArgs>
GradBoosting<WeakLearnerType, MatType>::GradBoosting(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const size_t num_models,
    WeakLearnerArgs&&... weakLearnerArgs) :
    num_models(num_models)
{
  WeakLearnerType other; // Will not be used.
  (void) TrainInternal<false>(data, labels, numClasses, other,
      weakLearnerArgs...);
}

// Train GradBoosting with a given weak learner.
template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>::Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models,
        const typename std::enable_if<
            std::is_same<WeakLearnerType, WeakLearnerInType>::value>::type* = 0
    )
{
    WeakLearnerType other; // Will not be used.
    return TrainInternal<false>(data, labels, numClasses, other);
}

template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>::Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const WeakLearnerInType& learner,
        const size_t num_models
    )
{
    return TrainInternal<true>(data, labels, numClasses, learner);
}


template<typename WeakLearnerType, typename MatType>
template<typename WeakLearnerInType>
typename MatType::elem_type GradBoosting<WeakLearnerType, MatType>::Train(
        const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t num_models,
        WeakLearnerArgs&&... weakLearnerArgs
    )
{
    WeakLearnerType other; // Will not be used.
    return TrainInternal<false>(data, labels, numClasses, other,
        weakLearnerArgs...);
}

// Classify the given test point.
template<typename WeakLearnerType, typename MatType>
template<typename VecType>
size_t GradBoosting<WeakLearnerType, MatType>::Classify(const VecType& point) const
{
    arma::Row<ElemType> probabilities;
    size_t prediction;
    Classify(point, prediction, probabilities);

    return prediction;
}

template<typename WeakLearnerType, typename MatType>
template<typename VecType>
void GradientBoosting<WeakLearnerType, MatType>::Classify(
    const VecType& point,
    size_t& prediction,
    arma::Row<typename MatType::elem_type>& probabilities) const
{
    probabilities.zeros(numClasses);
    
    // Aggregate predictions of each weak learner.
    for (size_t i = 0; i < numModels; ++i)
    {
        // Predict the residual using the weak learner.
        typename MatType::elem_type residual;
        wl[i].Predict(point, residual);
        
        // Add the prediction to the ensemble.
        probabilities += residual;
    }

    // Convert the ensemble predictions to probabilities.
    probabilities -= min(probabilities);
    probabilities /= accu(probabilities);

    // Determine the class with maximum probability as the final prediction.
    arma::uword maxIndex = 0;
    probabilities.max(maxIndex);
    prediction = (size_t) maxIndex;
}



}