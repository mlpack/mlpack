/**
 * @file methods/logistic_regression/logistic_regression_function_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegressionFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression_function.hpp"

#include <mlpack/core.hpp>

namespace mlpack {

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    const MatType& predictorsIn,
    const arma::Row<size_t>& responsesIn,
    const double lambda) :
    lambda(lambda)
{
  // We promise to be well-behaved... the elements won't be modified.
  MakeAlias(this->predictors, predictorsIn, predictorsIn.n_rows,
      predictorsIn.n_cols, 0, false);
  MakeAlias(this->responses, responsesIn, responsesIn.n_elem, 0, false);

  // Sanity check.
  if (responses.n_elem != predictors.n_cols)
  {
    Log::Fatal << "LogisticRegressionFunction::LogisticRegressionFunction(): "
        << "predictors matrix has " << predictors.n_cols << " points, but "
        << "responses vector has " << responses.n_elem << " elements (should be"
        << " " << predictors.n_cols << ")!" << std::endl;
  }
}

/**
 * Shuffle the datapoints.
 */
template<typename MatType>
void LogisticRegressionFunction<MatType>::Shuffle()
{
  MatType newPredictors;
  arma::Row<size_t> newResponses;

  ShuffleData(predictors, responses, newPredictors, newResponses);

  // If we are an alias, make sure we don't write to the original data.
  ClearAlias(predictors);
  ClearAlias(responses);

  // Take ownership of the new data.
  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters.
 */
template<typename MatType>
template<typename CoordinatesType>
typename CoordinatesType::elem_type
LogisticRegressionFunction<MatType>::Evaluate(
    const CoordinatesType& parameters) const
{
  // The objective function is the log-likelihood function (w is the parameters
  // vector for the model; y is the responses; x is the predictors; sig() is the
  // sigmoid function):
  //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
  // We want to minimize this function.  L2-regularization is just lambda
  // multiplied by the squared l2-norm of the parameters then divided by two.
  using ElemType = typename CoordinatesType::elem_type;

  // Specifying these here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType half = ((ElemType) 0.5);
  constexpr ElemType one = ((ElemType) 1);
  constexpr ElemType two = ((ElemType) 2);

  // For the regularization, we ignore the first term, which is the intercept
  // term and take every term except the last one in the decision variable.
  const ElemType regularization = half * lambda *
      dot(parameters.tail_cols(parameters.n_elem - 1),
      parameters.tail_cols(parameters.n_elem - 1));

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const CoordinatesType sigmoid = one / (one + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.  Note that the conversion causes some
  // copy and slowdown, but this is so negligible compared to the rest of the
  // calculation it is not worth optimizing for.
  const ElemType result = accu(log(one -
      ConvTo<CoordinatesType>::From(responses) + sigmoid %
      (two * ConvTo<CoordinatesType>::From(responses) - one)));

  // Invert the result, because it's a minimization.
  return regularization - result;
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters for a given batch from a given point.
 */
template<typename MatType>
template<typename CoordinatesType>
typename CoordinatesType::elem_type
LogisticRegressionFunction<MatType>::Evaluate(
    const CoordinatesType& parameters,
    const size_t begin,
    const size_t batchSize) const
{
  using ElemType = typename CoordinatesType::elem_type;

  // Specifying these here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType one = ((ElemType) 1);
  constexpr ElemType two = ((ElemType) 2);

  // Calculate the regularization term.
  const ElemType regularization = lambda *
      (batchSize / (two * predictors.n_cols)) *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const CoordinatesType sigmoid = one / (one + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1))));

  // Compute the objective for the given batch size from a given point.
  CoordinatesType respD = ConvTo<CoordinatesType>::From(
      responses.subvec(begin, begin + batchSize - 1));
  const ElemType result = accu(log(one - respD + sigmoid %
      (two * respD - one)));

  // Invert the result, because it's a minimization.
  return regularization - result;
}

//! Evaluate the gradient of the logistic regression objective function.
template<typename MatType>
template<typename CoordinatesType, typename GradType>
void LogisticRegressionFunction<MatType>::Gradient(
    const CoordinatesType& parameters,
    GradType& gradient) const
{
  using ElemType = typename CoordinatesType::elem_type;

  // Regularization term.
  GradType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  // Specifying this here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType one = ((ElemType) 1);

  const CoordinatesType sigmoids = (one / (one + exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(size(parameters));
  gradient[0] = -accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
      predictors.t() + regularization;
}

//! Evaluate the gradient of the logistic regression objective function for a
//! given batch size.
template<typename MatType>
template<typename CoordinatesType, typename GradType>
void LogisticRegressionFunction<MatType>::Gradient(
                const CoordinatesType& parameters,
                const size_t begin,
                GradType& gradient,
                const size_t batchSize) const
{
  using ElemType = typename CoordinatesType::elem_type;

  // Regularization term.
  GradType regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1)
      / predictors.n_cols * batchSize;

  // Specifying this here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType one = ((ElemType) 1);

  const CoordinatesType exponents = parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1);
  // Calculating the sigmoid function values.
  const CoordinatesType sigmoids = one / (one + exp(-exponents));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = -accu(responses.subvec(begin, begin + batchSize - 1) -
      sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids -
      responses.subvec(begin, begin + batchSize - 1)) *
      predictors.cols(begin, begin + batchSize - 1).t() + regularization;
}

/**
 * Evaluate the partial gradient of the logistic regression objective
 * function with respect to the individual features in the parameter.
 */
template<typename MatType>
template<typename CoordinatesType, typename GradType>
void LogisticRegressionFunction<MatType>::PartialGradient(
    const CoordinatesType& parameters,
    const size_t j,
    GradType& gradient) const
{
  using ElemType = typename CoordinatesType::elem_type;

  // Specifying this here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType one = ((ElemType) 1);

  const CoordinatesType diffs = responses - (one / (one + exp(-parameters(0, 0)
      - parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(size(parameters));

  if (j == 0)
  {
    gradient[j] = -accu(diffs);
  }
  else
  {
    gradient[j] = dot(-predictors.row(j - 1), diffs) + lambda *
        parameters(0, j);
  }
}

template<typename MatType>
template<typename CoordinatesType, typename GradType>
typename CoordinatesType::elem_type
LogisticRegressionFunction<MatType>::EvaluateWithGradient(
    const CoordinatesType& parameters,
    GradType& gradient) const
{
  using ElemType = typename CoordinatesType::elem_type;

  // Specifying these here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType one = ((ElemType) 1);
  constexpr ElemType two = ((ElemType) 2);

  // Regularization term.
  GradType regularization = lambda *
      parameters.tail_cols(parameters.n_elem - 1);

  const ElemType objectiveRegularization = lambda / two *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const CoordinatesType sigmoids = one / (one + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) * predictors)));

  gradient.set_size(size(parameters));
  gradient[0] = -accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids - responses) *
      predictors.t() + regularization;

  // Now compute the objective function using the sigmoids.
  ElemType result = accu(log(one -
      ConvTo<CoordinatesType>::From(responses) + sigmoids %
      (two * ConvTo<CoordinatesType>::From(responses) - one)));

  // Invert the result, because it's a minimization.
  return objectiveRegularization - result;
}

template<typename MatType>
template<typename CoordinatesType, typename GradType>
typename CoordinatesType::elem_type
LogisticRegressionFunction<MatType>::EvaluateWithGradient(
    const CoordinatesType& parameters,
    const size_t begin,
    GradType& gradient,
    const size_t batchSize) const
{
  using ElemType = typename CoordinatesType::elem_type;

  // Specifying these here makes the code below a little bit cleaner, and avoids
  // accidentally casting an entire expression to `double`, e.g., by the use of
  // `1.0` or similar.
  constexpr ElemType one = ((ElemType) 1);
  constexpr ElemType two = ((ElemType) 2);

  // Regularization term.
  GradType regularization =
      lambda * parameters.tail_cols(parameters.n_elem - 1) / predictors.n_cols *
      batchSize;

  const ElemType objectiveRegularization = lambda *
      (batchSize / (two * predictors.n_cols)) *
      dot(parameters.tail_cols(parameters.n_elem - 1),
          parameters.tail_cols(parameters.n_elem - 1));

  // Calculate the sigmoid function values.
  const CoordinatesType sigmoids = one / (one + exp(-(parameters(0, 0) +
      parameters.tail_cols(parameters.n_elem - 1) *
      predictors.cols(begin, begin + batchSize - 1))));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] = -accu(responses.subvec(begin, begin + batchSize - 1) -
      sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) = (sigmoids -
      responses.subvec(begin, begin + batchSize - 1)) *
      predictors.cols(begin, begin + batchSize - 1).t() + regularization;

  // Now compute the objective function using the sigmoids.
  CoordinatesType respD = ConvTo<CoordinatesType>::From(
      responses.subvec(begin, begin + batchSize - 1));
  const ElemType result = accu(log(one - respD + sigmoids %
      (two * respD - one)));

  // Invert the result, because it's a minimization.
  return objectiveRegularization - result;
}

} // namespace mlpack

#endif
