/**
 * @file methods/time_series_models/holt_winters.hpp
 * @author Suvarsha Chennareddy
 *
 * Definition of the Holt Winters Model for time series data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_HOLT_WINTERS_MODEL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_HOLT_WINTERS_MODEL_HPP

#include <mlpack/prereqs.hpp>
#include <ensmallen.hpp>

namespace mlpack {
namespace ts /* Time Series methods. */ {

/**
* The Holt-Winters method is a popular and effective approach 
* to forecasting seasonal time series. But different 
* methods will give different forecasts, 
* depending on whether the method is multiplicative or additive and 
* how the smoothing parameters are selected.
*/
template<char method = 'A', typename DataType = arma::Row<double>>
class HoltWintersModel
{
public:
/**
* Creates the model.
*/
HoltWintersModel();
/**
* Creates the model.
* @param Data The input data.
* @param seasonal_period The seasonal period m.
*
*/
HoltWintersModel(DataType data, const size_t  seasonal_period);

/**
* Creates the model.
* @param Data The input data.
* @param seasonal_period The seasonal period m.
* @param alpha  Smoothing parameter for level estimates.
* @param beta  Smoothing parameter for trend estimates.
* @param gamma  Smoothing parameter for seasonal estimates.
*/
HoltWintersModel(
    DataType data, const size_t seasonal_period, 
    double alpha, double beta, double gamma);

 /**
 * Trains the model based on the Data by updating the smoothing parameters.
 */
 void Train();

 /**
* This overload can be used to compute the
* mean squared error.
*/

 double Evaluate();


 /**
* This overload can be used to compute the
* mean squared error with the given data.
* @param Data The input data.
*/

 double Evaluate(const DataType& data);



 /**
 * This overload can be used to compute the
 * mean squared error with the given parameters.
 * @param params Smoothing parameters
 */

 double Evaluate(const arma::mat& params);

/**
* This overload can be used to make predictions on the
* dataset.
* @param predictions To store the predictions made on input data.
* @param H The number of time steps into the future to forcast.
*/
 void Predict(arma::Row<double>& predictions, size_t  hMax);


 /**
* This overload can be used to make predictions on the given
* dataset.
* @param Data The input data.
* @param predictions To store the predictions made on input data.
* @param H The number of time steps into the future to forcast.
*/
 void Predict(const DataType& data, arma::Row<double>& predictions, size_t  hMax);

/**
* This method can be used to make reset (override) the data.
* @param Data The input data.
* @param seasonal_period The seasonal period m.
*/
 void ResetData(DataType data, const size_t  seasonal_period);


 //! Get the level smoothing parameter.
 double const& alpha() const { return smoothingParams(0); }

 //! Get the trend smoothing parameter.
 double const& beta() const { return smoothingParams(1); }

 //! Get the seasonal smoothing parameter.
 double const& gamma() const { return smoothingParams(2); }


 //! Modify the level smoothing parameter.
 double& alpha() { return smoothingParams(0); }

 //! Modify the trend smoothing parameter.
 double&  beta() { return smoothingParams(1); }

 //! Modify the seasonal smoothing parameter.
 double& gamma() { return smoothingParams(2); }

private:


/**
* This overload can be used to compute the
* mean squared error with the given parameters and data.
* @param params Smoothing parameters
* @param Data The input data.
*/

double Evaluate(const arma::mat& params, const DataType& data);

/**
* This overload can be used to make predictions on the 
* dataset wtih the additive method.
* @param data Input data.
* @param predictions To store the predictions made on input data.
* @param alpha  Smoothing parameter for level estimates.
* @param beta  Smoothing parameter for trend estimates.
* @param gamma  Smoothing parameter for seasonal estimates.
* @param H The number of time steps into the future to forcast.
*/
template<char _method,
        typename _DataType,
        typename = std::enable_if_t<((arma::is_Row<_DataType>::value ||
            arma::is_Col<_DataType>::value) && (_method == 'A'))>>
 void Predict(const DataType& data,
     arma::Row<double>& predictions,
            double alpha,
            double beta,
            double gamma,
            size_t H = 0);

/**
* This overload can be used to make predictions on  dataset with the multiplicative method.
* @param data Input data.
* @param predictions To store the predictions made on input data.
* @param alpha  Smoothing parameter for level estimates.
* @param beta  Smoothing parameter for trend estimates.
* @param gamma  Smoothing parameter for seasonal estimates.
* @param H The number of time steps into the future to forcast.
*/
template<char _method,
    typename _DataType,
    typename = std::enable_if_t<((arma::is_Row<_DataType>::value ||
        arma::is_Col<_DataType>::value) && (_method == 'M'))>,
    typename = void>
 void Predict(const DataType& data,
     arma::Row<double>& predictions,
        double alpha,
        double beta,
        double gamma,
        size_t H = 0);

/**
* This method is used to initialize the level and trend vectors.
* @param data Input data.
* @param L Vector to store the level estimates
* @param B  Vector to store the trend estimates.
*/
void Initialize(const DataType& data, arma::Row<double>& L, arma::Row<double>& B);


//! Variable to store the Seasonal period parameter.
size_t  m;

//! Vector  to store the Data.
DataType Data;

//! Matrix (vector) to store the smoothing paramters of the model
arma::mat smoothingParams;

}; // class HoltWintersModel

} // namespace ts
} // namespace mlpack

// Include implementation.
#include "holt_winters_impl.hpp"

#endif