/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file phi.hpp
 *
 * This file computes the Gaussian probability
 * density function
 */
#include <mlpack/core.h>

namespace mlpack {
namespace gmm {

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
long double phi(const arma::vec& x,
                const arma::vec& mean,
                const arma::mat& cov) {
  long double cdet, f;
  double exponent;
  size_t dim;
  arma::mat cinv = inv(cov);
  arma::vec diff, tmp;

  dim = x.n_elem;
  cdet = det(cov);

  if (cdet < 0)
    cdet = -cdet;

  diff = mean - x;
  tmp = cinv * diff;
  exponent = dot(diff, tmp);

  long double tmp1, tmp2;
  tmp1 = 1 / pow(2 * M_PI, dim / 2);
  tmp2 = 1 / sqrt(cdet);
  f = (tmp1 * tmp2 * exp(-exponent / 2));

  return f;
}

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
long double phi(const double x, const double mean, const double var) {
  return exp(-1.0 * ((x - mean) * (x - mean) / (2 * var)))
      / sqrt(2 * M_PI * var);
}

/**
 * Calculates the multivariate Gaussian probability density function
 * and also the gradients with respect to the mean and the variance
 *
 * Example use:
 * @code
 * Vector x, mean, g_mean, g_cov;
 * ArrayList<Matrix> d_cov; // the dSigma
 * ....
 * long double f = phi(x, mean, cov, d_cov, &g_mean, &g_cov);
 * @endcode
 */
long double phi(const arma::vec& x,
                const arma::vec& mean,
                const arma::mat& cov,
                const std::vector<arma::mat>& d_cov,
                arma::vec& g_mean,
                arma::vec& g_cov) {
  long double cdet, f;
  double exponent;
  size_t dim;
  arma::mat cinv = inv(cov);
  arma::vec diff, tmp;

  dim = x.n_elem;
  cdet = det(cov);

  if (cdet < 0)
    cdet = -cdet;

  diff = mean - x;
  tmp = cinv * diff;
  exponent = dot(diff, tmp);

  long double tmp1, tmp2;
  tmp1 = 1 / pow(2 * M_PI, dim / 2);
  tmp2 = 1 / sqrt(cdet);
  f = (tmp1 * tmp2 * exp(-exponent / 2));

  // Calculating the g_mean values  which would be a (1 X dim) vector
  g_mean = f * tmp;

  // Calculating the g_cov values which would be a (1 X (dim*(dim+1)/2)) vector
  arma::vec g_cov_tmp(d_cov.size());
  for (size_t i = 0; i < d_cov.size(); i++) {
    arma::vec tmp_d;
    arma::mat inv_d;
    long double tmp_d_cov_d_r;

    tmp_d = d_cov[i] * tmp;
    tmp_d_cov_d_r = dot(tmp_d,tmp);
    inv_d = cinv * d_cov[i];
    for (size_t j = 0; j < dim; j++)
      tmp_d_cov_d_r += inv_d(j, j);
    g_cov_tmp[i] = f * tmp_d_cov_d_r / 2;
  }

  g_cov = g_cov_tmp;

  return f;
}

}; // namespace gmm
}; // namespace mlpack
