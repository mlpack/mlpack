/**
 * @file utils.hpp
 * @ _____
 *
 * Definition of some usefull function for preprocess the data
**/

#ifndef TATON_UTILS_HPP 
#define  TATON_UTILS_HPP

#include <mlpack/prereqs.hpp>

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


  
#endif
