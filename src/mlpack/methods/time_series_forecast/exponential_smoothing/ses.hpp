/**
 * @file methods/time_series/exponential_smoothing/ses.hpp
 * @author Aditi Pandey
 *
 * Single exponential smoothing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_EXPONENTIAL_SMOOTHING_SES_HPP
#define MLPACK_METHODS_EXPONENTIAL_SMOOTHING_SES_HPP

#include <mlpack/prereqs.hpp> 
#include <armadillo>

namespace mlpack {
namespace ts /** Single Exponential smoothing method of time series. */ {

/**
 * Single Exponential Smoothing, SES for short, also called
 * Simple Exponential Smoothing, is a time series forecasting
 * method for univariate data without a trend or seasonality.
 */

// function for calculating mean absolute error
double calc_mae(const arma::rowvec& errors); 
// function for calculating mean squared error
double calc_mse(const arma::rowvec& errors); 
// function for calculating mean absolute percentage error
double calc_mape(const arma::rowvec& data, const arma::rowvec& errors); 

class SingleES
{
 //forecasts for past periods
 arma::rowvec forecastPast; 
 //errors of forecasting past periods
 arma::rowvec errorForecastPast; 
 //the smoothing parameter
 double alpha; 
 
 public:
  /**
   * Creates the model.
   *
   * @param data, usually demand which varies with time.
   * @param alpha, the constant which determine the smoothing of the data points.
   */
  SingleES(arma::rowvec& data, const double& alpha);
  //Below is used to fetch the forecast vector
  arma::rowvec& GetForecastVector();
  //Below is used to fetch the error matrix w.r.t to forecast and the actual predictions
  arma::rowvec& GetErrorVector();
  //Below to get the smoothing parameter alpha
  double Getalpha() const;
  //Below to get the overall summary of the object created and trained
  void SesForecastSummary(arma::rowvec& data, SingleES& sesObject);

};

} // namespace ts
} // namespace mlpack

#endif // MLPACK_METHODS_EXPONENTIAL_SMOOTHING_SES_HPP
