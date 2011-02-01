/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file phi.h
 *
 * This file computes the Gaussian probability
 * density function
 */

#ifndef MLPACK_PHI_H
#define MLPACK_PHI_H

#include "fastlib/fastlib.h"
#include "fastlib/fastlib_int.h"
#include <cmath>

/**
 * Calculates the multivariate Gaussian probability density function
 * 
 * Example use:
 * @code
 * arma::vec x, mean;
 * arma::mat cov;
 * ....
 * long double f = phi(x, mean, cov);
 * @endcode
 */

long double phi(const arma::vec& x, const arma::vec& mean, const arma::mat& cov); 

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

long double phi(const double x, const double mean, const double var);

/**
 * Calculates the multivariate Gaussian probability density function 
 * and also the gradients with respect to the mean and the variance
 *
 * Example use:
 * @code
 * arma::vec x, mean, g_mean, g_cov;
 * std::vector<arma::mat> d_cov; // the dSigma
 * ....
 * long double f = phi(x, mean, cov, d_cov, &g_mean, &g_cov);
 * @endcode
 */

long double phi(const arma::vec& x, const arma::vec& mean, const arma::mat& cov, const std::vector<arma::mat>& d_cov, arma::vec& g_mean, arma::vec& g_cov);

#endif // MLPACK_PHI_H
