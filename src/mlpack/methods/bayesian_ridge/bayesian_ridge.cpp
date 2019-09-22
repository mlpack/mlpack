/**
 * @file bayesian_ridge.cpp
 * @author Clement Mercier 
 *
 * Implementation of Bayesian Ridge regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "bayesian_ridge.hpp"
#include <mlpack/core/util/log.hpp>
#include <mlpack/core/util/timers.hpp>

using namespace mlpack;
using namespace mlpack::regression;


BayesianRidge::BayesianRidge(const bool fitIntercept,
			     const bool normalize) :
  fitIntercept(fitIntercept),
  normalize(normalize){

  Log::Info << "Baysian Ridge regression(fitIntercept="
  	    << this->fitIntercept
  	    <<", normalize="
  	    <<this->normalize
  	    <<")"
  	    <<std::endl;
  }

void BayesianRidge::Train(const arma::mat& data,
			  const arma::rowvec& responses)
{
  Timer::Start("bayesian_ridge_regression");

  arma::mat phi;
  arma::rowvec t;
  arma::colvec vecphitT;
  arma::mat phiphiT;
  arma::colvec eigval;
  arma::mat eigvec;
  arma::colvec eigvali;
    
  // Preprocess the data. Center and normalize.
  this->CenterNormalize(data,
			responses,
			this->fitIntercept,
			this->normalize,
			phi,
			t,
			this->data_offset,
			this->data_scale,
			this->responses_offset);
  
  vecphitT = phi * t.t();
  phiphiT =  phi * phi.t();

  // Compute the eigenvalues only once.
  arma::eig_sym(eigval, eigvec, phiphiT);
   
  unsigned short p = data.n_rows, n = data.n_cols;
  // Initialize the hyperparameters and
  // begin with an infinitely broad prior.
  this->alpha = 1e-6;
  this->beta =  1 / (var(t) * 0.1);

  double tol = 1e-3; 
  unsigned short nIterMax = 50;
  unsigned short i = 0;
  double deltaAlpha = 1, deltaBeta = 1, crit = 1;
  arma::mat matA = arma::eye<arma::mat>(p, p);
  arma::rowvec temp;

  while ((crit > tol) && (i < nIterMax))
    {
      deltaAlpha = -this->alpha;
      deltaBeta = -this->beta;

      // Compute the posterior statistics.
      // with inv()
      for (size_t k = 0; k < p; k++) {matA(k,k) = this->alpha;}
      // inv is used instead of solve beacause we need matCovariance to
      // compute the prediction uncertainties. If solve is used, matCovariance
      // must be comptuted at the end of the loop.
      this->matCovariance = inv_sympd(matA + phiphiT * this->beta);
      this->omega = (this->matCovariance * vecphitT) * this->beta;

      // // with solve()
      // for (size_t k = 0; k < p; k++) {matA(k,k) = this->alpha / this->beta;}
      // this->omega = solve(matA + phiphiT, vecphitT);

      // Update alpha.
      eigvali = eigval * this->beta;
      gamma = sum(eigvali / (this->alpha + eigvali));
      this->alpha = gamma / dot(this->omega.t(), this->omega);

      // Update beta.
      temp = t - this->omega.t() * phi;
      this->beta = (n - gamma) / dot(temp, temp);
      
      // Comptute the stopping criterion.
      deltaAlpha += this->alpha;
      deltaBeta += this->beta;
      crit = abs(deltaAlpha/this->alpha + deltaBeta/this->beta);
      i++;
    }
  Timer::Stop("bayesian_ridge_regression");
}

void BayesianRidge::Predict(const arma::mat& points,
			    arma::rowvec& predictions) const
{
  arma::mat X = points;

  //Center and normalize the points before applying the model
  X.each_col() -= this->data_offset;
  X.each_col() /= this->data_scale;
  predictions = this->omega.t() * X + this->responses_offset; 
}

void BayesianRidge::Predict(const arma::colvec& point, double& prediction) const
{
  arma::mat point_mat = arma::conv_to<arma::mat>::from(point);
  arma::rowvec prediction_vec(1);
  this->Predict(point_mat, prediction_vec);
  prediction = prediction_vec[0];
}


void BayesianRidge::Predict(const arma::colvec& point,
			    double& prediction,
			    double& std) const
{
  arma::mat point_mat = arma::conv_to<arma::mat>::from(point);
  arma::rowvec prediction_vec(1);
  arma::rowvec std_vec(1);
  this->Predict(point_mat, prediction_vec, std_vec);
  prediction = prediction_vec[0];
  std = std_vec[0];
}

void BayesianRidge::Predict(const arma::mat& points,
			    arma::rowvec& predictions,
			    arma::rowvec& std) const
{
  arma::mat X = points;

  //Center and normalize the points before applying the model
  X.each_col() -= this->data_offset;
  X.each_col() /= this->data_scale;
  predictions = this->omega.t() * X + this->responses_offset;
  
  //Compute the standard deviation of each prediction
  std = arma::zeros<arma::rowvec>(X.n_cols);
  arma::colvec phi(X.n_rows);
  for (size_t i = 0; i < X.n_cols; i++)
    {
      phi = X.col(i);
      std[i] = sqrt(this->getVariance()
		    + dot(phi.t() * this->matCovariance, phi));
    }
}

double BayesianRidge::Rmse(const arma::mat& data,
			   const arma::rowvec& responses) const
{
  arma::rowvec predictions;
  this->Predict(data, predictions);
  return sqrt(mean(square(responses - predictions)));
}

void BayesianRidge::CenterNormalize(const arma::mat& data,
				    const arma::rowvec& responses,
				    bool fit_intercept,
				    bool normalize,
				    arma::mat& data_proc,
				    arma::rowvec& responses_proc,
				    arma::colvec& data_offset,
				    arma::colvec& data_scale,
				    double& responses_offset)
{
  // Initialize the offsets to their neutral forms.
  data_offset = arma::zeros<arma::colvec>(data.n_rows);
  data_scale = arma::ones<arma::colvec>(data.n_rows);
  responses_offset = 0.0;

  if (fit_intercept)
    {
      data_offset = mean(data, 1);
      responses_offset = mean(responses);
    }
  if (normalize)
      data_scale = stddev(data, 0, 1);

  // Copy data and response before the processing.
  data_proc = data;
  responses_proc = responses;
  // Center the data.
  data_proc.each_col() -= data_offset;
  // Scale the data.
  data_proc.each_col() /= data_scale;
  // Center the responses.
  responses_proc -= responses_offset;
}


// Copy construcor
BayesianRidge::BayesianRidge(const BayesianRidge& other):
  fitIntercept(other.fitIntercept),
  normalize(other.normalize),
  data_offset(other.data_offset),
  data_scale(other.data_scale),
  responses_offset(other.responses_offset),
  alpha(other.alpha),
  beta(other.beta),
  gamma(other.gamma),
  omega(other.omega),
  matCovariance(other.matCovariance)
{/* All is done */}

// Move construcor
BayesianRidge::BayesianRidge(BayesianRidge&& other):
  fitIntercept(other.fitIntercept),
  normalize(other.normalize),
  data_offset(std::move(other.data_offset)),
  data_scale(std::move(other.data_scale)),
  responses_offset(other.responses_offset),
  alpha(other.alpha),
  beta(other.beta),
  gamma(other.gamma),
  omega(std::move(other.omega)),
  matCovariance(std::move(other.matCovariance))
{
  // Clear the other object
  if (this != &other)
    {
      other.fitIntercept = false;
      other.normalize = false;
      other.data_offset.reset();
      other.data_scale.reset();
      other.responses_offset = 0.0;
      other.alpha = 0.0;
      other.gamma = 0.0;
      other.beta = 0.0;
      other.omega.reset();
      other.matCovariance.reset();
    }
}

BayesianRidge& BayesianRidge::operator=(const BayesianRidge& other)
{
  if (this == &other)
    return *this;
  
  fitIntercept = other.fitIntercept;
  normalize = other.normalize;
  data_offset = other.data_offset;
  data_scale = other.data_scale;
  responses_offset = other.responses_offset;
  alpha = other.alpha;
  gamma = other.gamma;
  beta = other.beta;
  omega = other.omega;
  matCovariance = other.matCovariance;
  return *this;
}

BayesianRidge& BayesianRidge::operator=(BayesianRidge&& other)
{
  if (this != &other )
    {
      fitIntercept = other.fitIntercept;
      normalize = other.normalize;
      data_offset = other.data_offset;
      data_scale = other.data_scale;
      responses_offset = other.responses_offset;
      alpha = other.alpha;
      gamma = other.gamma;
      beta = other.beta;
      omega = other.omega;
      matCovariance = other.matCovariance;
      
      // Clear the other object.
      other.fitIntercept = false;
      other.normalize = false;
      other.data_offset.reset();
      other.data_scale.reset();
      other.responses_offset = 0.0;
      other.alpha = 0.0;
      other.gamma = 0.0;
      other.beta = 0.0;
      other.omega.reset();
      other.matCovariance.reset();
    }
  return *this;
} 


  

