/**
 * @file softmax_regression_impl.hpp
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
namespace regression {

template<typename OptimizerType>
SoftmaxRegression::SoftmaxRegression(
    const arma::mat& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const bool fitIntercept,
    OptimizerType optimizer) :
    numClasses(numClasses),
    lambda(lambda),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, optimizer);
}

template<typename OptimizerType, typename... CallbackTypes>
SoftmaxRegression::SoftmaxRegression(
    const arma::mat& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const double lambda,
    const bool fitIntercept,
    OptimizerType optimizer,
    CallbackTypes&&... callbacks) :
    numClasses(numClasses),
    lambda(lambda),
    fitIntercept(fitIntercept)
{
  Train(data, labels, numClasses, optimizer, callbacks...);
}

template<typename VecType>
size_t SoftmaxRegression::Classify(const VecType& point) const
{
  arma::Row<size_t> label(1);
  Classify(point, label);
  return size_t(label(0));
}

template<typename OptimizerType>
double SoftmaxRegression::Train(const arma::mat& data,
                                const arma::Row<size_t>& labels,
                                const size_t numClasses,
                                OptimizerType optimizer)
{
  SoftmaxRegressionFunction regressor(data, labels, numClasses, lambda,
                                      fitIntercept);
  if (parameters.is_empty())
    parameters = regressor.GetInitialPoint();

  // Train the model.
  Timer::Start("softmax_regression_optimization");
  const double out = optimizer.Optimize(regressor, parameters);
  Timer::Stop("softmax_regression_optimization");

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

template<typename OptimizerType, typename... CallbackTypes>
double SoftmaxRegression::Train(const arma::mat& data,
                                const arma::Row<size_t>& labels,
                                const size_t numClasses,
                                OptimizerType optimizer,
                                CallbackTypes&&... callbacks)
{
  SoftmaxRegressionFunction regressor(data, labels, numClasses, lambda,
                                      fitIntercept);
  if (parameters.is_empty())
    parameters = regressor.GetInitialPoint();

  // Train the model.
  Timer::Start("softmax_regression_optimization");
  const double out = optimizer.Optimize(regressor, parameters, callbacks...);
  Timer::Stop("softmax_regression_optimization");

  Log::Info << "SoftmaxRegression::SoftmaxRegression(): final objective of "
            << "trained model is " << out << "." << std::endl;

  return out;
}

} // namespace regression
} // namespace mlpack

#endif
