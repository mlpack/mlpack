/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file phi.h
 *
 * This file computes the Gaussian probability
 * density function
 */
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

long double phi(arma::vec& x , arma::vec& mean , arma::mat& cov) {
	
  long double det, f;
  double exponent;
  index_t dim;
  arma::mat inv;
  arma::vec diff, tmp;
	
    dim = x.n_rows;
  inv = arma::inv(cov);
  det = arma::det(cov);
  
  if( det < 0){
    det = -det;
  }
  diff = mean - x;
  tmp = inv*diff;
  exponent = arma::dot(diff,tmp);
  long double tmp1, tmp2, tmp3;
  tmp1 = 1;
  tmp2 = dim;
  tmp2 = tmp2/2;
  tmp2 = pow((2*(math::PI)),tmp2);
  tmp1 = tmp1/tmp2;
  tmp3 = 1;
  tmp2 = sqrt(det);
  tmp3 = tmp3/tmp2;
  tmp2 = -exponent;
  tmp2 = tmp2 / 2;
  f = (tmp1*tmp3*exp(tmp2));

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

long double phi(double x, double mean, double var) {

  long double f;
	
  f = exp(-1.0*((x-mean)*(x-mean)/(2*var)))/sqrt(2*math::PI*var);
  return f;
}

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

long double phi(arma::vec& x, arma::vec& mean, arma::mat& cov, std::vector<arma::mat>& d_cov, arma::vec *g_mean, arma::vec *g_cov){
	
  long double det, f;
  double exponent;
  index_t dim;
  arma::mat inv;
  arma::vec diff, tmp;
	
  dim = x.n_rows;
  inv = arma::inv(cov);
  inv = arma::inv(cov);
  det = arma::det(cov); 

  if( det < 0){
    det = -det;
  }
  diff = mean - x;
  tmp = inv*diff;
  exponent = arma::dot(diff,tmp);
  long double tmp1, tmp2, tmp3;
  tmp1 = 1;
  tmp2 = dim;
  tmp2 = tmp2/2;
  tmp2 = pow((2*(math::PI)),tmp2);
  tmp1 = tmp1/tmp2;
  tmp3 = 1;
  tmp2 = sqrt(det);
  tmp3 = tmp3/tmp2;
  tmp2 = -exponent;
  tmp2 = tmp2 / 2;
  f = (tmp1*tmp3*exp(tmp2));

  // Calculating the g_mean values  which would be a (1 X dim) vector
  *g_mean = f*tmp;
	
  // Calculating the g_cov values which would be a (1 X (dim*(dim+1)/2)) vector
  arma::vec g_cov_tmp(d_cov.size());
  for(index_t i = 0; i < d_cov.size(); i++){
    arma::vec tmp_d;
    arma::mat inv_d;
    long double tmp_d_cov_d_r;
		
    tmp_d = d_cov[i]*tmp;
    tmp_d_cov_d_r = arma::dot(tmp_d,tmp);
    inv_d = inv*d_cov[i];
    for(index_t j = 0; j < dim; j++)
      tmp_d_cov_d_r += inv_d(j,j);
    g_cov_tmp[i] = f*tmp_d_cov_d_r/2;
  }
  *g_cov = g_cov_tmp; // Is g_cov larger than g_cov_tmp??
//  g_cov->Copy(g_cov_tmp,d_cov.size());
	
  return f;
}
