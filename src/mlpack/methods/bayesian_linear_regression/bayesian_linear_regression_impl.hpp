/**
 * @file methods/bayesian_linear_regression/bayesian_linear_regression_impl.hpp
 * @author Clement Mercier
 *
 * Implementation of templated BayesianLinearRegression functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_BAYESIAN_LINEAR_REGRESSION_IMPL_HPP

#include "bayesian_linear_regression.hpp"

namespace mlpack {

inline BayesianLinearRegression::BayesianLinearRegression(
    const bool centerData,
    const bool scaleData,
    const size_t maxIterations,
    const double tolerance) :
    centerData(centerData),
    scaleData(scaleData),
    maxIterations(maxIterations),
    tolerance(tolerance),
    responsesOffset(0.0),
    alpha(0.0),
    beta(0.0),
    gamma(0.0)
{/* Nothing to do */}

inline double BayesianLinearRegression::Train(const arma::mat& data,
                                              const arma::rowvec& responses)
{
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
  double crit = 1.0;

  while ((crit > tolerance) && (i < maxIterations))
  {
    double deltaAlpha = -alpha;
    double deltaBeta = -beta;

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

  return RMSE(data, responses);
}

inline void BayesianLinearRegression::Predict(const arma::mat& points,
                                              arma::rowvec& predictions) const
{
  // Center and scale the points before applying the model.
  arma::mat matX;
  CenterScaleDataPred(points, matX);
  predictions = omega.t() * matX + responsesOffset;
}

inline void BayesianLinearRegression::Predict(const arma::mat& points,
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

inline double BayesianLinearRegression::RMSE(
    const arma::mat& data,
    const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  Predict(data, predictions);
  return sqrt(mean(square(responses - predictions)));
}

inline double BayesianLinearRegression::CenterScaleData(
    const arma::mat& data,
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

inline void BayesianLinearRegression::CenterScaleDataPred(
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

/**
 * Serialize the Bayesian linear regression model.
 */
template<typename Archive>
void BayesianLinearRegression::serialize(Archive& ar,
                                         const uint32_t /* version */)
{
  ar(CEREAL_NVP(centerData));
  ar(CEREAL_NVP(scaleData));
  ar(CEREAL_NVP(maxIterations));
  ar(CEREAL_NVP(tolerance));
  ar(CEREAL_NVP(dataOffset));
  ar(CEREAL_NVP(dataScale));
  ar(CEREAL_NVP(responsesOffset));
  ar(CEREAL_NVP(alpha));
  ar(CEREAL_NVP(beta));
  ar(CEREAL_NVP(gamma));
  ar(CEREAL_NVP(omega));
  ar(CEREAL_NVP(matCovariance));
}

} // namespace mlpack

#endif
