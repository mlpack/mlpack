/**
 * @file methods/softmax_regression/softmax_regression_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of softmax regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_SOFTMAX_REGRESSION_SOFTMAX_REGRESSION_IMPL_HPP

// In case it hasn't been included yet.
#include "softmax_regression.hpp"

namespace mlpack {

template<typename MatType>
inline SoftmaxRegression<MatType>::SoftmaxRegression(
    const size_t inputSize,
    const size_t numClasses,
    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(0.0001),
    fitIntercept(fitIntercept)
{
  SoftmaxRegressionFunction<MatType>::InitializeWeights(
      parameters, inputSize, numClasses, fitIntercept);
}

template<typename MatType>
template<typename OptimizerType, typename... CallbackTypes, typename, typename>
SoftmaxRegression<MatType>::SoftmaxRegression(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const bool fitIntercept,
    CallbackTypes&&... callbacks) :
    numClasses(numClasses),
    lambda(lambda),
    fitIntercept(fitIntercept)
{
  OptimizerType optimizer;
  Train(data, labels, numClasses, lambda, fitIntercept, std::move(optimizer),
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename MatType>
template<typename OptimizerType, typename... CallbackTypes, typename, typename>
SoftmaxRegression<MatType>::SoftmaxRegression(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    OptimizerType& optimizer,
    const double lambda,
    const bool fitIntercept,
    CallbackTypes&&... callbacks) :
    numClasses(numClasses),
    lambda(lambda),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, lambda, fitIntercept, std::move(optimizer),
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename MatType>
template<typename OptimizerType,
         typename FirstCallbackType,
         typename... CallbackTypes,
         typename, typename, typename>
double SoftmaxRegression<MatType>::Train(const MatType& data,
                                         const arma::Row<size_t>& labels,
                                         const size_t numClasses,
                                         OptimizerType optimizer,
                                         FirstCallbackType&& firstCallback,
                                         CallbackTypes&&... callbacks)
{
  SoftmaxRegressionFunction<MatType> regressor(data, labels, numClasses, lambda,
                                               fitIntercept);
  if (parameters.n_elem != regressor.GetInitialPoint().n_elem)
    parameters = regressor.GetInitialPoint();

  // Train the model.
  const double out = optimizer.Optimize(regressor, parameters, firstCallback,
      callbacks...);
  this->numClasses = numClasses;

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

template<typename MatType>
template<typename OptimizerType, typename... CallbackTypes, typename, typename>
typename SoftmaxRegression<MatType>::ElemType
SoftmaxRegression<MatType>::Train(const MatType& data,
                                  const arma::Row<size_t>& labels,
                                  const size_t numClasses,
                                  const double lambda,
                                  const bool fitIntercept,
                                  CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(data, labels, numClasses, optimizer, lambda, fitIntercept,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename MatType>
template<typename OptimizerType, typename... CallbackTypes, typename, typename>
typename SoftmaxRegression<MatType>::ElemType
SoftmaxRegression<MatType>::Train(const MatType& data,
                                  const arma::Row<size_t>& labels,
                                  const size_t numClasses,
                                  OptimizerType& optimizer,
                                  const double lambda,
                                  const bool fitIntercept,
                                  CallbackTypes&&... callbacks)
{
  this->lambda = lambda;
  this->fitIntercept = fitIntercept;
  this->numClasses = numClasses;

  SoftmaxRegressionFunction<MatType> regressor(data, labels, numClasses, lambda,
                                               fitIntercept);
  if (parameters.n_elem != regressor.GetInitialPoint().n_elem)
    parameters = regressor.GetInitialPoint();

  // Train the model.
  const double out = optimizer.Optimize(regressor, parameters, callbacks...);
  this->numClasses = numClasses;

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

template<typename MatType>
inline void SoftmaxRegression<MatType>::Classify(const MatType& dataset,
                                                 arma::Row<size_t>& labels)
    const
{
  MatType probabilities;
  Classify(dataset, labels, probabilities);
}

template<typename MatType>
template<typename VecType>
size_t SoftmaxRegression<MatType>::Classify(const VecType& point) const
{
  arma::Row<size_t> label(1);
  Classify(point, label);
  return size_t(label(0));
}

template<typename MatType>
template<typename VecType>
void SoftmaxRegression<MatType>::Classify(
    const VecType& point,
    size_t& prediction,
    typename SoftmaxRegression<MatType>::DenseColType& probabilitiesVec) const
{
  arma::Row<size_t> label(1);
  Classify(point, label, probabilitiesVec);
  prediction = label[0];
}

template<typename MatType>
inline void SoftmaxRegression<MatType>::Classify(
    const MatType& dataset,
    arma::Row<size_t>& labels,
    typename SoftmaxRegression<MatType>::DenseMatType& probabilities) const
{
  util::CheckSameDimensionality(dataset, FeatureSize(),
      "SoftmaxRegression::Classify()");

  // Calculate the probabilities for each test input.
  DenseMatType hypothesis;
  if (fitIntercept)
  {
    // In order to add the intercept term, we should compute following matrix:
    //     [1; data] = join_cols(ones(1, data.n_cols), data)
    //     hypothesis = exp(parameters * [1; data]).
    //
    // Since the cost of join maybe high due to the copy of original data,
    // split the hypothesis computation to two components.
    hypothesis = exp(
      repmat(parameters.col(0), 1, dataset.n_cols) +
      parameters.cols(1, parameters.n_cols - 1) * dataset);
  }
  else
  {
    hypothesis = exp(parameters * dataset);
  }

  probabilities = hypothesis / repmat(sum(hypothesis, 0), numClasses, 1);

  // Prepare necessary data.
  labels.zeros(dataset.n_cols);
  double maxProbability = 0;

  // For each test input.
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    // For each class.
    for (size_t j = 0; j < numClasses; ++j)
    {
      // If a higher class probability is encountered, change prediction.
      if (probabilities(j, i) > maxProbability)
      {
        maxProbability = probabilities(j, i);
        labels(i) = j;
      }
    }

    // Set maximum probability to zero for the next input.
    maxProbability = 0;
  }
}

template<typename MatType>
inline void SoftmaxRegression<MatType>::Classify(
    const MatType& dataset,
    typename SoftmaxRegression<MatType>::DenseMatType& probabilities) const
{
  arma::Row<size_t> labels;
  Classify(dataset, labels, probabilities);
}

template<typename MatType>
inline double SoftmaxRegression<MatType>::ComputeAccuracy(
    const MatType& testData,
    const arma::Row<size_t>& labels) const
{
  arma::Row<size_t> predictions;

  // Get predictions for the provided data.
  Classify(testData, predictions);

  // Increment count for every correctly predicted label.
  size_t count = 0;
  for (size_t i = 0; i < predictions.n_elem; ++i)
    if (predictions(i) == labels(i))
      count++;

  // Return percentage accuracy.
  return (count * 100.0) / predictions.n_elem;
}

template<typename MatType>
void SoftmaxRegression<MatType>::Reset()
{
  SoftmaxRegressionFunction<MatType>::InitializeWeights(
      parameters, parameters.n_cols, parameters.n_rows, fitIntercept);
}

template<typename MatType>
template<typename Archive>
void SoftmaxRegression<MatType>::serialize(Archive& ar,
                                           const uint32_t version)
{
  if (cereal::is_loading<Archive>() && version == 0)
  {
    // This is the legacy version: `parameters` is of type arma::mat.
    arma::mat parametersTmp;
    ar(cereal::make_nvp("parameters", parametersTmp));
    parameters = ConvTo<DenseMatType>::From(parametersTmp);

    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(lambda));
    ar(CEREAL_NVP(fitIntercept));
  }
  else
  {
    ar(CEREAL_NVP(parameters));
    ar(CEREAL_NVP(numClasses));
    ar(CEREAL_NVP(lambda));
    ar(CEREAL_NVP(fitIntercept));
  }
}

} // namespace mlpack

#endif
