/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file phi.hpp
 *
 * This file computes the Gaussian probability
 * density function
 */
#ifndef __MLPACK_METHODS_MOG_PHI_HPP
#define __MLPACK_METHODS_MOG_PHI_HPP

#include <mlpack/core.h>

namespace mlpack {
namespace gmm {

/**
 * Calculates the univariate Gaussian probability density function
 *
 * Example use:
 * @code
 * double x, mean, var;
 * ....
 * long double f = phi(x, mean, var);
 * @endcode
 */
inline long double phi(const double x, const double mean, const double var)
{
  return exp(-1.0 * ((x - mean) * (x - mean) / (2 * var)))
      / sqrt(2 * M_PI * var);
}

/**
 * Calculates the multivariate Gaussian probability density function
 *
 * Example use:
 * @code
 * Vector x, mean;
 * Matrix cov;
 * ....
 * long double f = phi(x, mean, cov);
 * @endcode
 */
inline long double phi(const arma::vec& x,
                       const arma::vec& mean,
                       const arma::mat& cov)
{
  arma::vec diff = mean - x;

  arma::vec exponent = -0.5 * trans(diff) * inv(cov) * diff;

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
 * Vector x, mean, g_mean, g_cov;
 * ArrayList<Matrix> d_cov; // the dSigma
 * ....
 * long double f = phi(x, mean, cov, d_cov, &g_mean, &g_cov);
 * @endcode
 */
inline long double phi(const arma::vec& x,
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
  arma::vec exponent = -0.5 * trans(diff) * inv(cov) * diff;

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
  arma::vec exponents(x.n_cols); // We will now fill this.
  for (size_t i = 0; i < x.n_cols; i++)
    exponents(i) = accu(diffs.col(i) % rhs.col(i));

  probabilities = pow(2 * M_PI, (double) mean.n_elem / -2.0) *
      pow(det(cov), -0.5) * exp(exponents);
}

}; // namespace gmm
}; // namespace mlpack

#endif
