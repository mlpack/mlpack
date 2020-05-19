/**
 * @file rvmr.cpp
 * @author Clement Mercier
 *
 * Implementation of the Relevance Vector Machine.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef TATON_RVMR_IMPL_HPP 
#define  TATON_RVMR_IMPL_HPP

#include "rvmr.hpp"

using namespace mlpack;
namespace rvmr {

template<typename KernelType>
RVMR<KernelType>::RVMR(const KernelType& kernel,
		       const bool fitIntercept,
		       const bool normalize) :

  kernel(kernel),
  fitIntercept(fitIntercept),
  normalize(normalize),
  ardRegression(false) {

  std::cout << "RVMR_kernel_mlpack(fitIntercept="
	    << this->fitIntercept
	    << ", normalize="
	    << this->normalize
	    << ")"
	    << std::endl;
  }

  template <typename KernelType>
  RVMR<KernelType>::RVMR(const bool fitIntercept,
			 const bool normalize) :
    fitIntercept(fitIntercept),
    normalize(normalize),
    kernel(kernel::LinearKernel()),
    ardRegression(true) {

    std::cout << "RVMR_ARD_Regresion(fitIntercept="
	      << this->fitIntercept
	      << ", normalize="
	      << this->normalize
	      << ")"
	      << std::endl;
  }

template<typename KernelType>
void RVMR<KernelType>::Train(const arma::mat& data,
			     const arma::rowvec& responses)
{
  arma::mat phi;
  arma::rowvec t;

  // Manage the kernel.
  if (this->ardRegression == false)
  {
    // We must keep the original training data for future predictions.
    this->phi = data;
    applyKernel(data, data, phi);
      
    //Preprocess the data. Center and normalize.
    preprocess_data(phi,
		    responses,
		    this->fitIntercept,
		    this->normalize,
		    phi,
		    t,
		    this->data_offset,
		    this->data_scale,
		    this->responses_offset);
  }

  else
  {
    //Preprocess the data. Center and normalize.
    preprocess_data(data,
		    responses,
		    this->fitIntercept,
		    this->normalize,
		    phi,
		    t,
		    this->data_offset,
		    this->data_scale,
		    this->responses_offset);
  }

  unsigned short p = phi.n_rows, n = phi.n_cols;
  // Initialize the hyperparameters and
  // begin with an infinitely broad prior.
  this->alpha_threshold = 1e4;
  this->alpha = arma::ones<arma::rowvec>(p) * 1e-6;
  this->beta =  1 / (arma::var(t) * 0.1);

  // Loop variables.
  double tol = 1e-5;
  double L = 1.0;
  double crit = 1.0;
  unsigned short nIterMax = 50;
  unsigned short i = 0;
  unsigned short ind_act;

  arma::rowvec gammai = arma::zeros<arma::rowvec>(p);
  arma::mat matA;
  arma::rowvec temp(n);
  arma::mat subPhi;
  // Initiaze a vector of all the indices from the first
  // to the last point.
  arma::uvec allCols(n);
  for (size_t i=0; i < n; i++) {allCols(i) = i;}
  
  while ((crit > tol) && (i < nIterMax))
    {
      crit = -L;
      activeSet = find(alpha < alpha_threshold);
      // Prune out the inactive basis function. This procedure speeds up
      // the algorithm.
      subPhi = phi.submat(activeSet, allCols);

      // Compute the posterior statistics.
      matA = diagmat(alpha.elem(activeSet));
      matCovariance = inv(matA
				+ (subPhi
				   * subPhi.t())
				* beta);

      this->omega = (matCovariance
		     * subPhi
		     * t.t()) * beta;
      
      // Update the alpha_i.
      for (size_t k=0; k<this->activeSet.size(); k++)
	{
	  ind_act = activeSet[k];
	  gammai(ind_act) = 1 - matCovariance(k, k) * alpha(ind_act);

	  alpha(ind_act) = gammai(ind_act)
	    / (omega(k) * omega(k));
	}
      
      // Update beta.
      temp = t -  omega.t() * subPhi;
      beta = (n - sum(gammai.elem(activeSet))) / dot(temp, temp);
      
      // Comptute the stopping criterion.
      L = norm(omega);
      crit = abs(crit + L) / L;
      i++;
    }
}


template<typename KernelType>
void RVMR<KernelType>::Predict(const arma::mat& points,
			       arma::rowvec& predictions) const
{
  arma::mat X;
  // Manage the kernel.
  if (this->ardRegression == false)
    applyKernel(this->phi, points, X);
  else
    X = points;
  
  arma::uvec allCols(X.n_cols);
  for (size_t i=0; i < X.n_cols; i++) {allCols[i] = i;}
  
  // Center and normalize the points before applying the model.
  X.each_col() -= this->data_offset;
  X.each_col() /= this->data_scale;
  predictions = this->omega.t() * X.submat(this->activeSet, allCols)
                + this->responses_offset;
}


template<typename KernelType>
void RVMR<KernelType>::Predict(const arma::mat& points,
			       arma::rowvec& predictions,
			       arma::rowvec& std) const
{
  arma::mat X;
  // Manage the kernel.
  if (this->ardRegression == false)
    applyKernel(this->phi, points, X);
  else
    X = points;
  
  arma::uvec allCols(X.n_cols);
  for (size_t i=0; i < X.n_cols; i++) {allCols[i] = i;}
  
  // Center and normalize the points before applying the model.
  X.each_col() -= this->data_offset;
  X.each_col() /= this->data_scale;
  predictions = this->omega.t() * X.submat(this->activeSet, allCols)
                + this->responses_offset;

  // Comptute the standard deviations
  arma::mat O(X.n_cols, X.n_cols);
  O = X.submat(this->activeSet, allCols).t()
    * this->matCovariance
    * X.submat(this->activeSet, allCols);
  std = sqrt(diagvec(1/this->beta + O).t());
}


template<typename KernelType>
double RVMR<KernelType>::Rmse(const arma::mat& data,
		 const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  this->Predict(data, predictions);
  return sqrt(
	      mean(
		  square(responses - predictions)));
}


template<typename KernelType>
arma::vec RVMR<KernelType>::getCoefs() const
{
  // Get the size of the full solution with the offset.
  arma::colvec coefs = arma::zeros<arma::colvec>(this->data_offset.size());
  // omega[i] = 0 for the inactive basis functions

  // Now reconstruct the full solution. 
  for (size_t i=0; i < this->activeSet.size(); i++)
    {
      coefs[this->activeSet[i]] = this->omega[i];
    }
  return coefs;
}


template<typename KernelType>
void RVMR<KernelType>::applyKernel(const arma::mat& X,
				   const arma::mat& Y,
				   arma::mat& gramMatrix) const {

  // Check if the dimensions are consistent.
  if (X.n_rows != Y.n_rows)
    {
      std::cout << "error gramm" << std::endl;
      throw std::invalid_argument("Number of features not consistent");
    }
  
  gramMatrix = arma::zeros<arma::mat>(X.n_cols, Y.n_cols);
  arma::colvec xi = arma::zeros<arma::colvec>(X.n_rows);
  arma::colvec yj = arma::zeros<arma::colvec>(X.n_rows);

  for (size_t i=0; i < X.n_cols; i++)
    {
      xi = X.col(i);
      for (size_t j=0; j < Y.n_cols; j++)
	{
	  yj = Y.col(j);
	  gramMatrix(i, j) = this->kernel.Evaluate(xi, yj);
	}
    }
}

} // namespace rvmr;
#endif
