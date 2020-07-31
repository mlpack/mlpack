/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression.cpp
 * @author Clement Mercier 
 *
 * Implementation of Bayesian linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "bayesian_linear_regression.hpp"
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/timers.hpp>

using namespace mlpack;
using namespace mlpack::regression;

BayesianLinearRegression::BayesianLinearRegression(const bool centerData,
                                                   const bool scaleData,
                                                   const size_t nIterMax,
                                                   const double tol) :
  centerData(centerData),
  scaleData(scaleData),
  nIterMax(nIterMax),
  tol(tol),
  responsesOffset(0.0),
  alpha(0.0),
  beta(0.0),
  gamma(0.0)
{/* Nothing to do */}

double BayesianLinearRegression::Train(const arma::mat& data,
                                       const arma::rowvec& responses)
{
  Timer::Start("bayesian_linear_regression");

  arma::mat phi;
  arma::rowvec t;
  arma::colvec eigVal;
  arma::mat eigVec;

  // Preprocess the data. Center and scale.
  responsesOffset = CenterScaleData(data, responses, phi, t);

  if (!arma::eig_sym(eigVal, eigVec, arma::symmatu(phi * phi.t())))
  {
    Log::Fatal << "BayesianLinearRegression::Train(): Eigendecomposition "
               << "of covariance failed!" << std::endl;
  }

  // Compute this quantities once and for all.
  const arma::mat eigVecInv = inv(eigVec);
  const arma::colvec eigVecInvPhitT = eigVecInv * phi * t.t();

  // Initialize the hyperparameters and begin with an infinitely broad prior.
  alpha = 1e-6;
  beta =  1 / (var(t, 1) * 0.1);

  unsigned short i = 0;
  double deltaAlpha = 1.0, deltaBeta = 1.0, crit = 1.0;

  while ((crit > tol) && (i < nIterMax))
  {
    deltaAlpha = -alpha;
    deltaBeta = -beta;

    // Update the solution.
    omega = eigVec * diagmat(1 / (eigVal + (alpha / beta))) * eigVecInvPhitT;

    // Update alpha.
    gamma = sum(eigVal / (alpha / beta + eigVal));
    alpha = gamma / dot(omega, omega);

    // Update beta.
    const arma::rowvec temp = t - omega.t() * phi;
    beta = (data.n_cols - gamma) / dot(temp, temp);

    // Compute the stopping criterion.
    deltaAlpha += alpha;
    deltaBeta += beta;
    crit = std::abs(deltaAlpha / alpha + deltaBeta / beta);
    i++;
  }
  // Compute the covariance matrix for the uncertainties later.
  matCovariance = eigVec * diagmat(1 / (beta * eigVal + alpha)) * eigVecInv;

  Timer::Stop("bayesian_linear_regression");

  return RMSE(data, responses);
}

void BayesianLinearRegression::Predict(const arma::mat& points,
                                       arma::rowvec& predictions) const
{
  // Center and scale the points before applying the model.
  arma::mat matX;
  CenterScaleDataPred(points, matX);
  predictions = omega.t() * matX + responsesOffset;
}

void BayesianLinearRegression::Predict(const arma::mat& points,
                                       arma::rowvec& predictions,
                                       arma::rowvec& std) const
{
  // Center and scale the points before applying the model.
  arma::mat matX;
  CenterScaleDataPred(points, matX);
  predictions = omega.t() * matX + responsesOffset;
  // Compute the standard deviation for each point.
  std = sqrt(Variance() + sum(matX % (matCovariance * matX), 0));
}

double BayesianLinearRegression::RMSE(const arma::mat& data,
                                      const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  Predict(data, predictions);
  return sqrt(mean(square(responses - predictions)));
}

double BayesianLinearRegression::CenterScaleData(const arma::mat& data,
                                                 const arma::rowvec& responses,
                                                 arma::mat& dataProc,
                                                 arma::rowvec& responsesProc)
{
  if (!centerData && !scaleData)
  {
    dataProc = arma::mat(const_cast<double*>(data.memptr()), data.n_rows,
                                             data.n_cols, false, true);
    responsesProc = arma::rowvec(const_cast<double*>(responses.memptr()),
                                                     responses.n_elem, false,
                                                     true);
  }

  else if (centerData && !scaleData)
  {
    dataOffset = mean(data, 1);
    responsesOffset = mean(responses);
    dataProc = data.each_col() - dataOffset;
    responsesProc = responses - responsesOffset;
  }

  else if (!centerData && scaleData)
  {
    dataScale = stddev(data, 0, 1);
    dataProc = data.each_col() / dataScale;
    responsesProc = arma::rowvec(const_cast<double*>(responses.memptr()),
                                                     responses.n_elem, false,
                                                     true);
  }

  else
  {
    dataOffset = mean(data, 1);
    dataScale = stddev(data, 0, 1);
    responsesOffset = mean(responses);
    dataProc = (data.each_col() - dataOffset).each_col() / dataScale;
    responsesProc = responses - responsesOffset;
  }
  return responsesOffset;
}

void BayesianLinearRegression::CenterScaleDataPred(
    const arma::mat& data,
    arma::mat& dataProc) const
{
  if (!centerData && !scaleData)
  {
    dataProc = arma::mat(const_cast<double*>(data.memptr()), data.n_rows,
                         data.n_cols, false, true);
  }

  else if (centerData && !scaleData)
  {
    dataProc = data.each_col() - dataOffset;
  }

  else if (!centerData && scaleData)
  {
    dataProc = data.each_col() / dataScale;
  }

  else
  {
    dataProc = (data.each_col() - dataOffset).each_col() / dataScale;
  }
}
