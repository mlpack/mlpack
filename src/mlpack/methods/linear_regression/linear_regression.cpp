/**
 * @file methods/linear_regression/linear_regression.cpp
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
#include "linear_regression.hpp"
#include <mlpack/core/util/log.hpp>

using namespace mlpack;
using namespace mlpack::regression;

LinearRegression::LinearRegression(const arma::mat& predictors,
                                   const arma::rowvec& responses,
                                   const double lambda,
                                   const bool intercept) :
    LinearRegression(predictors, responses, arma::rowvec(), lambda, intercept)
{}

LinearRegression::LinearRegression(const arma::mat& predictors,
                                   const arma::rowvec& responses,
                                   const arma::rowvec& weights,
                                   const double lambda,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  Train(predictors, responses, weights, intercept);
}

double LinearRegression::Train(const arma::mat& predictors,
                               const arma::rowvec& responses,
                               const bool intercept)
{
  return Train(predictors, responses, arma::rowvec(), intercept);
}

double LinearRegression::Train(const arma::mat& predictors,
                               const arma::rowvec& responses,
                               const arma::rowvec& weights,
                               const bool intercept)
{
  this->intercept = intercept;

  /*
   * We want to calculate the a_i coefficients of:
   * \sum_{i=0}^n (a_i * x_i^i)
   * In order to get the intercept value, we will add a row of ones.
   */

  // We store the number of rows and columns of the predictors.
  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).
  const size_t nCols = predictors.n_cols;

  arma::mat p = predictors;
  arma::rowvec r = responses;

  // Here we add the row of ones to the predictors.
  // The intercept is not penalized. Add an "all ones" row to design and set
  // intercept = false to get a penalized intercept.
  if (intercept)
  {
    p.insert_rows(0, arma::ones<arma::mat>(1, nCols));
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
  arma::mat cov = p * p.t() +
      lambda * arma::eye<arma::mat>(p.n_rows, p.n_rows);

  parameters = arma::solve(cov, p * r.t());
  return ComputeError(predictors, responses);
}

void LinearRegression::Predict(const arma::mat& points,
    arma::rowvec& predictions) const
{
  if (intercept)
  {
    // We want to be sure we have the correct number of dimensions in the
    // dataset.
    Log::Assert(points.n_rows == parameters.n_rows - 1);
    // Get the predictions, but this ignores the intercept value
    // (parameters[0]).
    predictions = arma::trans(parameters.subvec(1, parameters.n_elem - 1))
        * points;
    // Now add the intercept.
    predictions += parameters(0);
  }
  else
  {
    // We want to be sure we have the correct number of dimensions in
    // the dataset.
    Log::Assert(points.n_rows == parameters.n_rows);
    predictions = arma::trans(parameters) * points;
  }
}

double LinearRegression::ComputeError(const arma::mat& predictors,
                                      const arma::rowvec& responses) const
{
  // Get the number of columns and rows of the dataset.
  const size_t nCols = predictors.n_cols;
  const size_t nRows = predictors.n_rows;

  // Calculate the differences between actual responses and predicted responses.
  // We must also add the intercept (parameters(0)) to the predictions.
  arma::rowvec temp;
  if (intercept)
  {
    // Ensure that we have the correct number of dimensions in the dataset.
    if (nRows != parameters.n_rows - 1)
    {
      Log::Fatal << "The test data must have the same number of columns as the "
          "training file." << std::endl;
    }
    temp = responses - (parameters(0) +
        arma::trans(parameters.subvec(1, parameters.n_elem - 1)) * predictors);
  }
  else
  {
    // Ensure that we have the correct number of dimensions in the dataset.
    if (nRows != parameters.n_rows)
    {
      Log::Fatal << "The test data must have the same number of columns as the "
          "training file." << std::endl;
    }
    temp = responses - arma::trans(parameters) * predictors;
  }
  const double cost = arma::dot(temp, temp) / nCols;

  return cost;
}
