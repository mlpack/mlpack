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
#include <type_traits>
#include "linear_regression.hpp"

namespace mlpack {
namespace regression {

template <template<class> class M, class Ele>
inline LinearRegressionModel<M, Ele>::LinearRegressionModel(
    const M<Ele>& predictors,
    const arma::Row<Ele>& responses,
    const double lambda,
    const bool intercept) :
    LinearRegressionModel(predictors,
		                  arma::Mat<Ele>(responses),
		                  arma::Row<Ele>(),
		                  lambda,
		                  intercept)
{ /* Nothing to do. */ }

template <template<class> class M, class Ele>
inline LinearRegressionModel<M, Ele>::LinearRegressionModel(
    const M<Ele>& predictors,
    const arma::Mat<Ele>& responses,
    const double lambda,
    const bool intercept) :
    LinearRegressionModel(predictors,
                          responses,
                          arma::Row<Ele>(),
                          lambda,
                          intercept)
{ /* Nothing to do. */ }

template <template<class> class M, class Ele>
inline LinearRegressionModel<M, Ele>::LinearRegressionModel(
    const M<Ele>& predictors,
    const arma::Row<Ele>& responses,
    const arma::Row<Ele>& weights,
    const double lambda,
    const bool intercept) :
    LinearRegressionModel(predictors,
                          arma::Mat<Ele>(responses),
                          weights,
                          lambda,
                          intercept)   
{ /* Nothing to do. */ }

template <template<class> class M, class Ele>
inline LinearRegressionModel<M, Ele>::LinearRegressionModel(
    const M<Ele>& predictors,
    const arma::Mat<Ele>& responses,
    const arma::Row<Ele>& weights,
    const double lambda,
    const bool intercept) : lambda(lambda), intercept(intercept)
{
  Train(predictors, responses, weights, intercept);
}

template <template<class> class M, class Ele>
inline double LinearRegressionModel<M, Ele>::Train(const M<Ele>& predictors,
                                      const arma::Row<Ele>& responses,
                                      const bool intercept)
{
  return Train(predictors,
               arma::Mat<Ele>(responses),
               arma::Row<Ele>(),
               intercept);
}

template <template<class> class M, class Ele>
inline double LinearRegressionModel<M, Ele>::Train(const M<Ele>& predictors,
               			              const arma::Row<Ele>& responses,
               			              const arma::Row<Ele>& weights,
               			              const bool intercept)
{
  return Train(predictors, arma::Mat<Ele>(responses), weights, intercept);
}

template <template<class> class M, class Ele>
inline double LinearRegressionModel<M, Ele>::Train(const M<Ele>& predictors,
                                      const arma::Mat<Ele>& responses,
                                      const bool intercept)
{
  return Train(predictors, responses, arma::Mat<Ele>(), intercept);
}

template <template<class> class M, class Ele>
inline double LinearRegressionModel<M, Ele>::Train(const M<Ele>& predictors,
                                      const arma::Mat<Ele>& responses,
                                      const arma::Row<Ele>& weights,
                                      const bool intercept)
{
  this->intercept = intercept;

  /*
   * train a multi-output regression model
   * In order to get the intercept value, we will add a row of ones.
   */

  // We store the number of rows and columns of the predictors.
  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).

  // Sanity check on data.
  util::CheckSameSizes(predictors,
                       responses,
                       "LinearRegressionModel::Train()");

  const int nCols = predictors.n_cols;

  M<Ele> p;
  arma::Mat<Ele> r = responses;

  // Here we add the row of ones to the predictors.
  // The intercept is not penalized. Add an "all ones" row to design and set
  // intercept = false to get a penalized intercept.
  if (intercept){
     p = M<Ele>(predictors.n_rows + 1, nCols);
     p.submat(1, 0, predictors.n_rows, nCols-1) = predictors;
     for (int i=0; i < nCols; i++) p(0, i) = 1;
  }
  else p = predictors;

  if (weights.n_elem > 0)
  {
    p = p * diagmat(sqrt(weights));
    r = responses * diagmat(sqrt(weights));
  }

  // Convert to this form:
  // a * (X X^T) = y X^T.
  // Then we'll use Armadillo to solve it.
  M<Ele> cov = p * p.t();
  cov.diag() += lambda;
  _solve(cov, p * r.t());
  return ComputeError(predictors, responses);
}

template <template<class> class M, class Ele>
inline void LinearRegressionModel<M, Ele>::Predict(const M<Ele>& points,
    				                  arma::Row<Ele>& predictions) const
{
  arma::Mat<Ele> preds_mat;
  Predict(points, preds_mat);
  predictions = arma::Row<Ele>(preds_mat);
}

template <template<class> class M, class Ele>
inline void LinearRegressionModel<M, Ele>::Predict(const M<Ele>& points,
    				                  arma::Mat<Ele>& predictions) const
{
  if (intercept)
  {
    // We want to be sure we have the correct number of dimensions in the
    // dataset.
    // Prevent underflow.
    const size_t labels = (parameters.n_rows == 0) ? size_t(0) :
        size_t(parameters.n_rows - 1);
    util::CheckSameDimensionality(points,
                                  labels,
                                  "LinearRegressionModel::Predict()", 
                                  "points");
    // Get the predictions, but this ignores the intercept value
    predictions = arma::trans(parameters.rows(1, parameters.n_rows - 1))
        * points;
    // Now add the intercept.
    // predictions.each_col() += parameters.row(0).t();
    for (size_t i=0; i < predictions.n_cols; i++) 
            predictions.col(i) += parameters.row(0).t(); 
  }
  else
  {
    // We want to be sure we have the correct number of dimensions in
    // the dataset.
    util::CheckSameDimensionality(points,
                                  parameters, 
                                  "LinearRegressionModel::Predict()",
                                  "points");
    predictions = arma::trans(parameters) * points;
  }
}

template <template<class> class M, class Ele>
inline double LinearRegressionModel<M, Ele>::ComputeError(
    const M<Ele>& predictors,
    const arma::Row<Ele>& responses) const
{
  return ComputeError(predictors, arma::Mat<Ele>(responses));
}

template <template<class> class M, class Ele>
inline double LinearRegressionModel<M, Ele>::ComputeError(
    const M<Ele>& predictors,
    const arma::Mat<Ele>& responses) const
{
  // Sanity check on data.
  util::CheckSameSizes(predictors,
                       responses,
                       "LinearRegressionModel::Train()");

  arma::Mat<Ele> diff;
  arma::Mat<Ele> preds;
  Predict(predictors, preds);
  diff = responses - preds; 
  return arma::accu(diff % diff) / diff.n_cols; 
}


} // namespace regression 
} // namespace mlpack

#endif
