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
#include <mlpack/methods/grad_boosting/grad_boosting.hpp>
#include <mlpack/core.hpp>

// Defined within the mlpack namespace.
namespace mlpack {

// Empty constructor.
template<typename WeakLearnerType, typename MatType>
GradBoosting<WeakLearnerType, MatType>::GradBoosting() :
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
template<typename WeakLearnerType, typename MatType>
GradBoosting<WeakLearnerType, MatType>::
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
template<typename WeakLearnerType, typename MatType>

// Variadic template to the Weak Learner arguments
template<typename... WeakLearnerArgs>
GradBoosting<WeakLearnerType, MatType>::
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

template<typename WeakLearnerType, typename MatType>
void GradBoosting<WeakLearnerType, MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels,
        const WeakLearnerType& learner)
{
  return TrainInternal<true>(data, labels, numModels, numClasses, learner);
}

template<typename WeakLearnerType, typename MatType>
void GradBoosting<WeakLearnerType, MatType>::
  Train(const MatType& data,
        const arma::Row<size_t>& labels,
        const size_t numClasses,
        const size_t numModels)
{
  WeakLearnerType other; // Will not be used.
  return TrainInternal<false>(data, labels, numModels, numClasses, other);
}

template<typename WeakLearnerType, typename MatType>
template<typename... WeakLearnerArgs>
void GradBoosting<WeakLearnerType, MatType>::
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
template<typename WeakLearnerType, typename MatType>
template<typename VecType>
size_t GradBoosting<WeakLearnerType, MatType>::Classify(const VecType& point) 
{
  size_t prediction;
  Classify(point, prediction);
  return prediction;
}

template<typename WeakLearnerType, typename MatType>
template<typename VecType>
void GradBoosting<WeakLearnerType, MatType>::Classify(const VecType& point,
                                                      size_t& prediction)
{

  for (size_t i = 0; i < weakLearners.size(); ++i) 
  {
    size_t tempPred = weakLearners[i].Classify(point);
    prediction += tempPred;
  }

}


template<typename WeakLearnerType, typename MatType>
void Classify(const MatType& test,
              arma::Row<size_t>& predictedLabels) 
{
  for (size_t i = 0; i < test.size(); ++i) {
    size_t prediction;
    Classify(test[i], prediction);
    predictedLabels[i] = prediction;
  }
}


template<
  bool UseExistingWeakLearner,
  typename MatType,
  typename WeightsType,
  typename WeakLearnerType,
  typename... WeakLearnerArgs
>
struct WLTrainer
{
  static WeakLearnerType Train(const MatType& data,
                                const arma::Row<size_t>& labels,
                                const size_t numClasses,
                                const WeightsType& weights,
                                const WeakLearnerType& wl,
                                WeakLearnerArgs&&... /*weakLearnerArgs*/)
  {
    // Use the existing weak learner to train a new one with new weights.
    // API requirement: there is a constructor with this signature:
    //
    //    WeakLearnerType(const WeakLearnerType&,
    //                    MatType& data,
    //                    LabelsType& labels,
    //                    const size_t numClasses)
    //
    // This trains the new WeakLearnerType using the hyperparameters from the
    // given WeakLearnerType.
    return WeakLearnerType(wl, data, labels, numClasses, weights);
  }
};

template<
    typename MatType,
    typename WeightsType,
    typename WeakLearnerType,
    typename... WeakLearnerArgs
>
struct WLTrainer<
    false, MatType, WeightsType, WeakLearnerType, WeakLearnerArgs...
>
{
  static WeakLearnerType Train(
      const MatType& data,
      const arma::Row<size_t>& labels,
      const size_t numClasses,
      const WeightsType& weights,
      const WeakLearnerType& /* wl */,
      WeakLearnerArgs&&... weakLearnerArgs)
  {
    // When UseExistingWeakLearner is false, we use the given hyperparameters.
    // (This is the preferred approach that supports more types of weak
    // learners.)
    return WeakLearnerType(data, labels, numClasses, weights,
        weakLearnerArgs...);
  }
};


struct DefaultWeakLearnerArgs
{
  // Define default values for the weak learner arguments here.
  static constexpr double defaultValue1 = 1.0;
  static constexpr int defaultValue2 = 42;
  static constexpr int defaultValue3 = 42;
};




// This function uses default arguments
template<typename WeakLearnerType, typename MatType>
template<bool UseExistingWeakLearner>
void GradBoosting<WeakLearnerType, MatType>::TrainInternal(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numModels,
    const size_t numClasses,
    const WeakLearnerType& wl)
{
  // Call the main TrainInternal function with default arguments
  return TrainInternal<UseExistingWeakLearner>(
      data, labels, numModels, numClasses, wl,
      DefaultWeakLearnerArgs::defaultValue1);
}


// Template for GradBoosting template as a whole
template<typename WeakLearnerType, typename MatType>

// Template for TrainInternal 
// UseExistingWeakLearner determines whether to define a weak learner anew or 
// use an existing weak learner
// WeakLearnerArgs are the arguments for the weak learner
template<bool UseExistingWeakLearner, typename... WeakLearnerArgs>

// TrainInternal is a private function within GradBoosting class
// It has return type ElemType
void GradBoosting<WeakLearnerType, MatType>:: 
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
  
  arma::Row<size_t> residue = labels; 

  for (size_t model = 0; model < numModels; ++model) 
  {

    // Build the weight vectors.
    weights = sum(D);

    WeakLearnerType w = WLTrainer<
        UseExistingWeakLearner, MatType, arma::Row<ElemType>, WeakLearnerType,
        WeakLearnerArgs...
    >::Train(data, residue, numClasses, weights, wl, weakLearnerArgs...);

    weakLearners.push_back(w);

    arma::Row<size_t> predictions = residue;
    w.Classify(data, predictions);

    residue = residue - predictions;
        
  }
}

}

#endif

