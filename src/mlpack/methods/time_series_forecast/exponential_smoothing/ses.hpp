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

namespace mlpack {
namespace single_exponential_smoothing /** Single Exponential smoothing method. */ {

/**
 * Single Exponential Smoothing, SES for short, also called
 * Simple Exponential Smoothing, is a time series forecasting
 * method for univariate data without a trend or seasonality.
 */

// function for calculating mean absolute error
double calc_mae(std::vector<double>&); 
// function for calculating mean squared error
double calc_mse(std::vector<double>&); 
// function for calculating mean absolute percentage error
double calc_mape(std::vector<double>&, std::vector<double>&); 

class SingleES
{
 //forecasts for past periods
 std::vector<double> forecast_past; 
 //errors of forecasting past periods
 std::vector<double> error_forecast_past; 
 //the smoothing parameter
 double alpha; 
 // Implementation of the single exponential smoothing
 void update(const double&, const double&);
 
 public:
  /**
   * Creates the model.
   *
   * @param data, usually demand which varies with time.
   * @param alpha, the constant which determine the smoothing of the data points.
   */
  SingleES(std::vector<double>& data, const double& alpha);
  //Below is used to fetch the forecast vector
  std::vector<double>& forecast_vector_ref();
  //Below is used to fetch the error matrix w.r.t to forecast and the actual predictions
  std::vector<double>& error_vector_ref();
  double getalpha() const;

};

} // namespace single_exponential_smoothing
} // namespace mlpack

#endif // MLPACK_METHODS_EXPONENTIAL_SMOOTHING_SES_HPP
