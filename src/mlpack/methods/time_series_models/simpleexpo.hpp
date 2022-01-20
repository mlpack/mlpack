/**
 * @file methods/time_series_methods/simpleexpo.hpp
 * @author Rishabh Bali
 *
 * Definition of the SimpleExpo class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_SIMPLEEXPO_MODEL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_SIMPLEEXPO_MODEL_HPP

#include<mlpack/prereqs.hpp>

namespace mlpack {
namespace ts {

/** 
 * Simple Exponential Smopothing mopdel is one of the simplest time series forcasting methods.
 * It simply gives more weights to recent observations while forcasting and the 
 * weights given to older observations exponentially decreases.
 **/ 
class SimpleExpo
{
 public :
 /**
   * Creates the model. and also initializes the parameters.
   *
   * @param data time series data taken as input 
   * @param alpha weighted parameter
   */

 SimpleExpo(const arma::mat & data, const double alpha);

 /**
   * Empty constructor.  This gives a non-working model, so make sure Train() is
   * called (or make sure the model parameters are set) before calling
   * Predict()!
   */

 SimpleExpo();
 /**
   * Creates the model. and also initializes the parameters.
   *Here aplha is not given by the user.
   * @param data time series data taken as input.
   */


 SimpleExpo(const arma::mat & data);

   /**
   * Train the SES model on the data to . trhis function will return the least squares error
   *
   * @param data  time series data to train the model on
   * @param alpha the parameter which gives weights to the different data points
   * @return The least squares error after training.
   */
 double Train(const arma::mat&data,const double alpha);

 /**
   * Calculate y_i for each data point in data.
   *
   * @param data time series data to predict the outcome
   */

 void Predict(const arma::mat& data,arma::rowvec& predictions) const ;


 double ComputeError(const arma::mat & data,arma::rowvec& predictions)const;

//! Return the parameter alpha
 const double Alpha() const {return alpha;}

//! Returns the initial level 
 const double Level() const {return level;}

//! Modifies the parameters alpha
 double& Alpha();

//! Modifies the initial level 
 double & Level(); 


 private :

 double alpha;

 double level;
 
 arma::rowvec predictions;
};

} // namespace ts
} //namespace mlpack

#endif    //MLPACK_METHODS_TIME_SERIES_MODEELS_SIMPLEEXPO_MODEL_HPP