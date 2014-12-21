/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file phi.hpp
 *
 * This file computes the Gaussian probability
 * density function
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_MOG_PHI_HPP
#define __MLPACK_METHODS_MOG_PHI_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace gmm {

/**
 * Calculates the univariate Gaussian probability density function.
 *
 * Example use:
 * @code
 * double x, mean, var;
 * ....
 * double f = phi(x, mean, var);
 * @endcode
 *
 * @param x Observation.
 * @param mean Mean of univariate Gaussian.
 * @param var Variance of univariate Gaussian.
 * @return Probability of x being observed from the given univariate Gaussian.
 */
inline double phi(const double x, const double mean, const double var)
{
  return exp(-1.0 * ((x - mean) * (x - mean) / (2 * var)))
      / sqrt(2 * M_PI * var);
}

/**
 * Calculates the multivariate Gaussian probability density function.
 *
 * Example use:
 * @code
 * extern arma::vec x, mean;
 * extern arma::mat cov;
 * ....
 * double f = phi(x, mean, cov);
 * @endcode
 *
 * @param x Observation.
 * @param mean Mean of multivariate Gaussian.
 * @param cov Covariance of multivariate Gaussian.
 * @return Probability of x being observed from the given multivariate Gaussian.
 */
inline double phi(const arma::vec& x,
                  const arma::vec& mean,
                  const arma::mat& cov)
{
  arma::vec diff = mean - x;

  // Parentheses required for Armadillo 3.0.0 bug.
  arma::vec exponent = -0.5 * (trans(diff) * inv(cov) * diff);

  // TODO: What if det(cov) < 0?
  return pow(2 * M_PI, (double) x.n_elem / -2.0) * pow(det(cov), -0.5) *
      exp(exponent[0]);
}

/**
 * Calculates the multivariate Gaussian probability density function and also
 * the gradients with respect to the mean and the variance.
 *
 * Example use:
 * @code
 * extern arma::vec x, mean, g_mean, g_cov;
 * std::vector<arma::mat> d_cov; // the dSigma
 * ....
 * double f = phi(x, mean, cov, d_cov, &g_mean, &g_cov);
 * @endcode
 */
inline double phi(const arma::vec& x,
                  const arma::vec& mean,
                  const arma::mat& cov,
                  const std::vector<arma::mat>& d_cov,
                  arma::vec& g_mean,
                  arma::vec& g_cov)
{
  // We don't call out to another version of the function to avoid inverting the
  // covariance matrix more than once.
  arma::mat cinv = inv(cov);

  arma::vec diff = mean - x;
  // Parentheses required for Armadillo 3.0.0 bug.
  arma::vec exponent = -0.5 * (trans(diff) * inv(cov) * diff);

  long double f = pow(2 * M_PI, (double) x.n_elem / 2) * pow(det(cov), -0.5)
      * exp(exponent[0]);

  // Calculate the g_mean values; this is a (1 x dim) vector.
  arma::vec invDiff = cinv * diff;
  g_mean = f * invDiff;

  // Calculate the g_cov values; this is a (1 x (dim * (dim + 1) / 2)) vector.
  for (size_t i = 0; i < d_cov.size(); i++)
  {
    arma::mat inv_d = cinv * d_cov[i];

    g_cov[i] = f * dot(d_cov[i] * invDiff, invDiff) +
        accu(inv_d.diag()) / 2;
  }

  return f;
}

/**
 * Calculates the multivariate Gaussian probability density function for each
 * data point (column) in the given matrix, with respect to the given mean and
 * variance.
 *
 * @param x List of observations.
 * @param mean Mean of multivariate Gaussian.
 * @param cov Covariance of multivariate Gaussian.
 * @param probabilities Output probabilities for each input observation.
 */
inline void phi(const arma::mat& x,
                const arma::vec& mean,
                const arma::mat& cov,
                arma::vec& probabilities)
{
  // Column i of 'diffs' is the difference between x.col(i) and the mean.
  arma::mat diffs = x - (mean * arma::ones<arma::rowvec>(x.n_cols));

  // Now, we only want to calculate the diagonal elements of (diffs' * cov^-1 *
  // diffs).  We just don't need any of the other elements.  We can calculate
  // the right hand part of the equation (instead of the left side) so that
  // later we are referencing columns, not rows -- that is faster.
  arma::mat rhs = -0.5 * inv(cov) * diffs;
  arma::vec exponents(diffs.n_cols); // We will now fill this.
  for (size_t i = 0; i < diffs.n_cols; i++)
    exponents(i) = exp(accu(diffs.unsafe_col(i) % rhs.unsafe_col(i)));

  probabilities = pow(2 * M_PI, (double) mean.n_elem / -2.0) *
      pow(det(cov), -0.5) * exponents;
}

}; // namespace gmm
}; // namespace mlpack

#endif
