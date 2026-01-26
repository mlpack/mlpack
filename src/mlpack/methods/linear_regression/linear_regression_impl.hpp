/**
 * @file methods/linear_regression/linear_regression_impl.hpp
 * @author James Cline
 * @author Michael Fox
 *
 * Implementation of simple linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_IMPL_HPP

#include "linear_regression.hpp"

namespace mlpack {

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename>
inline LinearRegression<ModelMatType>::LinearRegression(
    const MatType& predictors,
    const ResponsesType& responses,
    const double lambda,
    const bool intercept) :
    LinearRegression(predictors, responses,
        arma::Row<typename ResponsesType::elem_type>(), lambda, intercept)
{ /* Nothing to do. */ }

template<typename ModelMatType>
template<typename MatType,
         typename ResponsesType,
         typename WeightsType,
         typename, typename>
inline LinearRegression<ModelMatType>::LinearRegression(
    const MatType& predictors,
    const ResponsesType& responses,
    const WeightsType& weights,
    const double lambda,
    const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  Train(predictors, responses, weights, lambda, intercept);
}

template<typename ModelMatType>
template<typename T>
inline double LinearRegression<ModelMatType>::Train(
    const arma::mat& predictors,
    const arma::rowvec& responses,
    const T intercept,
    const std::enable_if_t<std::is_same_v<T, bool>>*)
{
  return Train(predictors, responses, arma::rowvec(), this->lambda, intercept);
}

template<typename ModelMatType>
template<typename T>
inline double LinearRegression<ModelMatType>::Train(
    const arma::mat& predictors,
    const arma::rowvec& responses,
    const arma::rowvec& weights,
    const T intercept,
    const std::enable_if_t<std::is_same_v<T, bool>>*)
{
  return Train(predictors, responses, weights, this->lambda, intercept);
}

template<typename ModelMatType>
template<typename MatType>
inline
typename LinearRegression<ModelMatType>::ElemType
LinearRegression<ModelMatType>::Train(const MatType& predictors,
                                      const arma::rowvec& responses)
{
  return Train(predictors, responses, arma::rowvec(), this->lambda,
      this->intercept);
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType, typename, typename>
inline
typename LinearRegression<ModelMatType>::ElemType
LinearRegression<ModelMatType>::Train(const MatType& predictors,
                                      const ResponsesType& responses,
                                      const std::optional<double> lambda,
                                      const std::optional<bool> intercept)
{
  return Train(predictors, responses,
      arma::Row<typename ResponsesType::elem_type>(),
      (lambda.has_value()) ? lambda.value() : this->lambda,
      (intercept.has_value()) ? intercept.value() : this->intercept);
}

template<typename ModelMatType>
template<typename MatType>
inline
typename LinearRegression<ModelMatType>::ElemType
LinearRegression<ModelMatType>::Train(const MatType& predictors,
                                      const arma::rowvec& responses,
                                      const arma::rowvec& weights)
{
  return Train(predictors, responses, weights, this->lambda, this->intercept);
}

template<typename ModelMatType>
template<typename MatType,
         typename ResponsesType,
         typename WeightsType,
         typename, typename>
inline
typename LinearRegression<ModelMatType>::ElemType
LinearRegression<ModelMatType>::Train(const MatType& predictors,
                                      const ResponsesType& responses,
                                      const WeightsType& weights,
                                      const std::optional<double> lambda,
                                      const std::optional<bool> intercept)
{
  if (lambda.has_value())
    this->lambda = lambda.value();

  if (intercept.has_value())
    this->intercept = intercept.value();

  /*
   * We want to calculate the a_i coefficients of:
   * \sum_{i=0}^n (a_i * x_i^i)
   * In order to get the intercept value, we will add a row of ones.
   */

  // We store the number of rows and columns of the predictors.
  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).

  // Sanity check on data.
  util::CheckSameSizes(predictors, responses, "LinearRegression::Train()");

  const size_t nCols = predictors.n_cols;

  // TODO: avoid copy if possible.
  arma::Mat<ElemType> p = ConvTo<arma::Mat<ElemType>>::From(predictors);
  arma::Row<ElemType> r = responses;

  // Here we add the row of ones to the predictors.
  // The intercept is not penalized. Add an "all ones" row to design and set
  // intercept = false to get a penalized intercept.
  if (this->intercept)
  {
    p.insert_rows(0, ones<arma::Mat<ElemType>>(1, nCols));
  }

  if (weights.n_elem > 0)
  {
    p = p * diagmat(sqrt(weights));
    r = sqrt(weights) % responses;
  }

  // Convert to this form:
  // a * (X X^T) = y X^T.
  // Then we'll use Armadillo to solve it.
  // The total runtime of this should be O(d^2 N) + O(d^3) + O(dN).
  // (assuming the SVD is used to solve it)
  arma::Mat<ElemType> cov = p * p.t() +
      ((ElemType) this->lambda) *
      arma::eye<arma::Mat<ElemType>>(p.n_rows, p.n_rows);

  parameters = arma::solve(cov, p * r.t());
  return ComputeError(predictors, responses);
}

template<typename ModelMatType>
template<typename VecType>
inline
typename LinearRegression<ModelMatType>::ElemType
LinearRegression<ModelMatType>::Predict(const VecType& point) const
{
  if (intercept)
  {
    // We want to be sure we have the correct number of dimensions in the
    // dataset.
    // Prevent underflow.
    const size_t dimensionality = (parameters.n_rows == 0) ? size_t(0) :
        size_t(parameters.n_rows - 1);
    util::CheckSameDimensionality(point, dimensionality,
        "LinearRegression::Predict()", "point");

    return dot(parameters.subvec(1, parameters.n_elem - 1).t(), point) +
        parameters(0);
  }
  else
  {
    // We want to be sure we have the correct number of dimensions in
    // the dataset.
    util::CheckSameDimensionality(point, parameters,
        "LinearRegression::Predict()", "point");

    return dot(parameters.t(), point);
  }
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType>
inline void LinearRegression<ModelMatType>::Predict(
    const MatType& points,
    ResponsesType& predictions) const
{
  if (intercept)
  {
    // We want to be sure we have the correct number of dimensions in the
    // dataset.
    // Prevent underflow.
    const size_t dimensionality = (parameters.n_rows == 0) ? size_t(0) :
        size_t(parameters.n_rows - 1);
    util::CheckSameDimensionality(points, dimensionality,
        "LinearRegression::Predict()", "points");

    // Get the predictions, but this ignores the intercept value
    // (parameters[0]).
    predictions = parameters.subvec(1, parameters.n_elem - 1).t() * points;
    // Now add the intercept.
    predictions += parameters(0);
  }
  else
  {
    // We want to be sure we have the correct number of dimensions in
    // the dataset.
    util::CheckSameDimensionality(points, parameters,
        "LinearRegression::Predict()", "points");
    predictions = trans(parameters) * points;
  }
}

template<typename ModelMatType>
template<typename MatType, typename ResponsesType>
inline typename LinearRegression<ModelMatType>::ElemType
LinearRegression<ModelMatType>::ComputeError(
    const MatType& predictors,
    const ResponsesType& responses) const
{
  // Sanity check on data.
  util::CheckSameSizes(predictors, responses, "LinearRegression::Train()");

  // Get the number of columns and rows of the dataset.
  const size_t nCols = predictors.n_cols;
  const size_t nRows = predictors.n_rows;

  // Calculate the differences between actual responses and predicted responses.
  // We must also add the intercept (parameters(0)) to the predictions.
  arma::Row<typename ResponsesType::elem_type> temp;
  if (intercept)
  {
    // Ensure that we have the correct number of dimensions in the dataset.
    if (nRows != parameters.n_rows - 1)
    {
      Log::Fatal << "The test data must have the same number of columns as the "
          "training file." << std::endl;
    }
    temp = responses - (parameters(0) +
        parameters.subvec(1, parameters.n_elem - 1).t() * predictors);
  }
  else
  {
    // Ensure that we have the correct number of dimensions in the dataset.
    if (nRows != parameters.n_rows)
    {
      Log::Fatal << "The test data must have the same number of columns as the "
          "training file." << std::endl;
    }
    temp = responses - parameters.t() * predictors;
  }
  const ElemType cost = dot(temp, temp) / nCols;

  return cost;
}

template<typename ModelMatType>
template<typename Archive>
void LinearRegression<ModelMatType>::serialize(Archive& ar,
                                               const uint32_t version)
{
  if (cereal::is_loading<Archive>() && version == 0)
  {
    // Old versions represented `parameters` as an arma::vec.
    arma::vec parametersTmp;
    ar(cereal::make_nvp("parameters", parametersTmp));
    parameters = ConvTo<ModelColType>::From(parametersTmp);
  }
  else
  {
    ar(CEREAL_NVP(parameters));
  }

  ar(CEREAL_NVP(lambda));
  ar(CEREAL_NVP(intercept));
}

} // namespace mlpack

#endif
