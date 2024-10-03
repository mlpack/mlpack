/**
 * @file methods/linear_svm/linear_svm_impl.hpp
 * @author Ayush Chamoli
 *
 * Implementation of Linear SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_IMPL_HPP
#define MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_IMPL_HPP

// In case it hasn't been included yet.
#include "linear_svm.hpp"

namespace mlpack {

template<typename ModelMatType>
LinearSVM<ModelMatType>::LinearSVM() :
    lambda(0.0001),
    delta(1.0),
    fitIntercept(false)
{
  // No training to do here.
}

template<typename ModelMatType>
LinearSVM<ModelMatType>::LinearSVM(
    const size_t dimensionality,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  LinearSVMFunction<ModelMatType /* fake, does not matter for this call */,
                    ModelMatType>::InitializeWeights(
      parameters, dimensionality, numClasses, fitIntercept);
}

template<typename ModelMatType>
template<typename OptimizerType, typename... CallbackTypes, typename, typename>
LinearSVM<ModelMatType>::LinearSVM(
    const arma::mat& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept,
    OptimizerType optimizer,
    CallbackTypes&&... callbacks) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, optimizer,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename ModelMatType>
template<typename OptimizerType, typename>
LinearSVM<ModelMatType>::LinearSVM(
    const arma::mat& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept,
    OptimizerType optimizer) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, optimizer);
}

template<typename ModelMatType>
template<typename MatType, typename... CallbackTypes, typename>
LinearSVM<ModelMatType>::LinearSVM(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const double delta,
    const bool fitIntercept,
    CallbackTypes&&... callbacks) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  // By default we use L-BFGS.
  ens::L_BFGS optimizer;
  Train(data, labels, numClasses, optimizer,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename ModelMatType>
template<typename MatType,
         typename OptimizerType,
         typename... CallbackTypes,
         typename>
LinearSVM<ModelMatType>::LinearSVM(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    OptimizerType& optimizer,
    const double lambda,
    const double delta,
    const bool fitIntercept,
    CallbackTypes&&... callbacks) :
    numClasses(numClasses),
    lambda(lambda),
    delta(delta),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, optimizer,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename ModelMatType>
template<typename MatType, typename... CallbackTypes, typename>
typename LinearSVM<ModelMatType>::ElemType LinearSVM<ModelMatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    CallbackTypes&&... callbacks)
{
  return Train(data, labels, numClasses, this->lambda, this->delta,
      this->fitIntercept, std::forward<CallbackTypes>(callbacks)...);
}

template<typename ModelMatType>
template<typename MatType, typename... CallbackTypes, typename>
typename LinearSVM<ModelMatType>::ElemType LinearSVM<ModelMatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const std::optional<double> delta,
    const std::optional<bool> fitIntercept,
    CallbackTypes&&... callbacks)
{
  // By default, train with L-BFGS.
  ens::L_BFGS lbfgs;
  return Train(data, labels, numClasses, lbfgs, lambda,
      (delta.has_value()) ? delta.value() : this->delta,
      (fitIntercept.has_value()) ? fitIntercept.value() : this->fitIntercept,
      std::forward<CallbackTypes>(callbacks)...);
}

template<typename ModelMatType>
template<typename MatType,
         typename OptimizerType,
         typename... CallbackTypes,
         typename, typename>
typename LinearSVM<ModelMatType>::ElemType LinearSVM<ModelMatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    OptimizerType optimizer,
    CallbackTypes&&... callbacks)
{
  return Train(data, labels, numClasses, optimizer, this->lambda, this->delta,
      this->fitIntercept, std::forward<CallbackTypes>(callbacks)...);
}

template<typename ModelMatType>
template<typename MatType,
         typename OptimizerType,
         typename... CallbackTypes,
         typename, typename>
typename LinearSVM<ModelMatType>::ElemType LinearSVM<ModelMatType>::Train(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    OptimizerType optimizer,
    const double lambda,
    const std::optional<double> delta,
    const std::optional<bool> fitIntercept,
    CallbackTypes&&... callbacks)
{
  this->numClasses = numClasses;
  this->lambda = lambda;

  if (delta.has_value())
    this->delta = delta.value();

  if (fitIntercept.has_value())
    this->fitIntercept = fitIntercept.value();

  if (numClasses <= 1)
  {
    throw std::invalid_argument("LinearSVM dataset has 0 number of classes!");
  }

  LinearSVMFunction<MatType, ModelMatType> svm(data, labels, numClasses,
      this->lambda, this->delta, this->fitIntercept);
  const bool needNewParameters = (parameters.is_empty() ||
      (this->fitIntercept && (parameters.n_rows != data.n_rows + 1)) ||
      (!this->fitIntercept && (parameters.n_rows != data.n_rows)) ||
      (parameters.n_cols != numClasses));
  if (needNewParameters)
    parameters = svm.InitialPoint();

  // Train the model.
  const ElemType out = optimizer.Optimize(svm, parameters, callbacks...);

  Log::Info << "LinearSVM::LinearSVM(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

template<typename ModelMatType>
template<typename MatType>
void LinearSVM<ModelMatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& labels) const
{
  DenseMatType scores;
  Classify(data, labels, scores);
}

template<typename ModelMatType>
template<typename MatType>
void LinearSVM<ModelMatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& labels,
    typename LinearSVM<ModelMatType>::DenseMatType& scores) const
{
  util::CheckSameDimensionality(data, FeatureSize(), "LinearSVM::Classify()");

  if (fitIntercept)
  {
    scores = parameters.rows(0, parameters.n_rows - 2).t() * data
        + repmat(parameters.row(parameters.n_rows - 1).t(), 1, data.n_cols);
  }
  else
  {
    scores = parameters.t() * data;
  }

  // Prepare necessary data.
  labels.zeros(data.n_cols);

  labels = ConvTo<arma::Row<size_t>>::From(
      arma::index_max(scores));
}

template<typename ModelMatType>
void LinearSVM<ModelMatType>::Classify(
    const arma::mat& data,
    arma::mat& scores) const
{
  util::CheckSameDimensionality(data, FeatureSize(), "LinearSVM::Classify()");

  if (fitIntercept)
  {
    scores = parameters.rows(0, parameters.n_rows - 2).t() * data
        + repmat(parameters.row(parameters.n_rows - 1).t(), 1, data.n_cols);
  }
  else
  {
    scores = parameters.t() * data;
  }
}

template<typename ModelMatType>
template<typename VecType>
size_t LinearSVM<ModelMatType>::Classify(const VecType& point) const
{
  arma::Row<size_t> label(1);
  Classify(point, label);
  return size_t(label(0));
}

template<typename ModelMatType>
template<typename VecType>
void LinearSVM<ModelMatType>::Classify(
    const VecType& point,
    size_t& label,
    typename LinearSVM<ModelMatType>::DenseColType& probabilities) const
{
  arma::Row<size_t> labelRow(1);
  Classify(point, labelRow, probabilities);
  label = labelRow[0];
}

template<typename ModelMatType>
template<typename MatType>
double LinearSVM<ModelMatType>::ComputeAccuracy(
    const MatType& testData,
    const arma::Row<size_t>& testLabels) const
{
  arma::Row<size_t> labels;

  // Get predictions for the provided data.
  Classify(testData, labels);

  // Increment count for every correctly predicted label.
  size_t count = 0;
  for (size_t i = 0; i < labels.n_elem ; ++i)
    if (testLabels(i) == labels(i))
      count++;

  // Return the accuracy.
  return (double) 100.0 * count / labels.n_elem;
}

template<typename ModelMatType>
template<typename Archive>
void LinearSVM<ModelMatType>::serialize(Archive& ar, const uint32_t version)
{
  // Old versions used `arma::mat` for the type of `parameters`.
  if (cereal::is_loading<Archive>() && version == 0)
  {
    arma::mat parametersTmp;
    ar(cereal::make_nvp("parameters", parametersTmp));
    parameters = ConvTo<arma::mat>::From(parametersTmp);
  }
  else
  {
    ar(CEREAL_NVP(parameters));
  }

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(lambda));
  ar(CEREAL_NVP(fitIntercept));
}

} // namespace mlpack

#endif // MLPACK_METHODS_LINEAR_SVM_LINEAR_SVM_IMPL_HPP
