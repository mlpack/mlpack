/**
 * @file robust_svd.cpp
 * @author Charan Reddy
 *
 * Implementation of the robust SVD method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "robust_svd.hpp"


namespace mlpack {
namespace svd {

RobustSVD::RobustSVD(const arma::mat& data,
                             arma::mat& u,
                             arma::vec& s,
                             arma::mat& v,
                             const size_t iteratedPower,
                             const size_t maxIterations,
                             const double eps) :
    iteratedPower(iteratedPower),
    maxIterations(maxIterations),
    eps(eps)
{
    Apply(data, u, s, v);
}

RobustSVD::RobustSVD(const size_t iteratedPower,
                             const size_t maxIterations,
                             const double eps) :
    iteratedPower(iteratedPower),
    maxIterations(maxIterations),
    eps(eps)
{
  /* Nothing to do here */
}


void RobustSVD::Apply(const arma::mat& data,
                          arma::mat& u,
                          arma::vec& s,
                          arma::mat& v)
{
  int M = data.n_rows, N = data.n_cols;
  arma::mat S = arma::zeros(M, N);
  arma::mat Y = arma::zeros(M, N);
  double mu = M*N/(4* arma::accu(arma::pow(data,2)));
  double mu_inv = 1/ mu;
  double lmbda = 1/std::sqrt(std::max(M,N)*1.0);
  int iter =0;
  double err = std::pow(10,18);
  arma::mat L = arma::zeros(M, N);
  double tol = (1e-7)* arma::accu(arma::pow(arma::abs(data),2));
  int max_iter = 1000;
  arma::mat X;
  arma::vec check1;
  arma::mat check2;
  while(err>tol && iter<max_iter)
  {
	X = data - S + mu_inv * Y;  
	arma::svd(u, s, v, X);
	check1 = arma::zeros(size(s));
	check1 = arma::sign(s) * arma::max((arma::abs(s) - mu_inv), check1);
	L = (u * (arma::diagmat(check1) * v));
	X = data - L + mu_inv * Y;
	check2 = arma::zeros(size(X));
	S = arma::sign(X) * arma::max((arma::abs(X) - mu_inv*lmbda), check2);
	Y = Y + mu * (data - L - S);
	err = arma::accu(arma::pow(arma::abs(data - L - S) , 2));
	iter+=1;
	if(iter%100 !=0 || iter ==1 || iter >max_iter || err<=tol)
	{
		std::cout<<"iteration: "<< iter << " error: "<<err<<std::endl;
	} 
  }

}

} // namespace svd
} // namespace mlpack
