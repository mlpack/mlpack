/**
 * @file utils.hpp
 * @ _____
 *
 * Definition of some usefull function for preprocess the data
**/

#ifndef TATON_UTILS_HPP 
#define  TATON_UTILS_HPP

#include <mlpack/prereqs.hpp>

typedef double (*kernel)(arma::mat, arma::mat, double);

/*
 * Center and normalize the data. The last four arguments
 * allow future modifation of new points.
 *
 * @param data Design matrix in column-major format, dim(P,N).
 * @param responses A vector of targets.
 * @param fit_interpept If true data will be centred according to the points.
 * @param fit_interpept If true data will be scales by the standard deviations
 *     of the features computed according to the points.
 * @param data_proc data processed, dim(N,P).
 * @param responses_proc responses processed, dim(N).
 * @param data_offset Mean vector of the design matrix according to the 
 *     points, dim(P).
 * @param data_scale Vector containg the standard deviations of the features
 *     dim(P).
 * @param reponses_offset Mean of responses.
 */
void preprocess_data(const arma::mat& data,
		     const arma::rowvec& responses,
		     const bool fit_intercept,
		     const bool normalize,
		     arma::mat& data_proc,
		     arma::rowvec& responses_proc,
		     arma::colvec& data_offset,
		     arma::colvec& data_scale,
		     double& responses_offset);

/*
 * Compute gram matrix between two matrices X and Y.
 *
 * @param X Matrice dim(p,n1).
 * @param Y Matrice dim(p,n2).
 * @param kernelFunction Function pointer toward a kernel function.
 *     Available : linear, rbf.  
 * @param gamma Length scale parameter of the rbf kernel.
 */
void gramMatrix(const arma::mat& X,
		const arma::mat& Y,
		arma::mat& gramMatrix,
		double (*kernelFunction)(arma::colvec&, arma::colvec&, double),
		double gamma);

/*
 * Compute the Radial Basis Function between two vectors.
 * 
 * @param x Vector.
 * @param y Vector.
 * @param gamma Length scale parameter of the rbf kernel. If gamma
 * @return rbf Value of the kernel function.
*/
double rbf(arma::colvec& x,
	   arma::colvec& y,
	   double gamma);

/*
 * Compute the linear kernel function between two vectors.
 * 
 * @param x Vector.
 * @param y Vector.
 * @param gamma Length scale parameter of the rbf kernel. If gamma
 * @return linear Value of the kernel function.
*/
double linear(arma::colvec& x,
	      arma::colvec& y,
	      double gamma);


  
#endif
