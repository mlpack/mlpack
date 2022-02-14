/**
 * @file methods/time_series_models/holt_winters_impl.hpp
 * @author Suvarsha Chennareddy
 *
 * Implementation of the Persistence Model for time series data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_HOLT_WINTERS_MODEL_IMPL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_HOLT_WINTERS_MODEL_IMPL_HPP

 // In case it hasn't yet been included.
#include "holt_winters.hpp"

namespace mlpack {
namespace ts /* Time Series methods. */ {
template<char method, typename DataType>
HoltWintersModel<method, DataType>::HoltWintersModel():
    m(1),
    smoothingParams({0.5, 0.5, 0.5}) //Initializing the smoothing parameters to 0.5
{ /* Nothing to do here. */}


template<char method, typename DataType>
HoltWintersModel<method, DataType>::HoltWintersModel(
    DataType data, const size_t seasonal_period):
    m(seasonal_period),
    Data(data),
    smoothingParams({0.5, 0.5, 0.5}) //Initializing the smoothing parameters to 0.5
{ /* Nothing to do here. */}

template<char method, typename DataType>
HoltWintersModel<method, DataType>::HoltWintersModel(
    DataType data, const size_t seasonal_period, 
    double alpha, double beta, double gamma) :
    m(seasonal_period),
    Data(data),
    smoothingParams({ alpha, beta, gamma })
{ /* Nothing to do here. */}

template<char method, typename DataType>
void HoltWintersModel<method, DataType>::Train()
{
    ens::SA<> optimizer; 
    optimizer.Optimize(*this, smoothingParams);
}
template<char method, typename DataType>
double  HoltWintersModel<method, DataType>::Evaluate() {
    return Evaluate(smoothingParams, Data);
}

template<char method, typename DataType>
double  HoltWintersModel<method, DataType>::Evaluate(
    const DataType& data) {
    return Evaluate(smoothingParams, data);
}

template<char method, typename DataType>
double  HoltWintersModel<method, DataType>::Evaluate(
    const arma::mat& params) {
    return Evaluate(params, Data);
}

 template<char method, typename DataType>
 double  HoltWintersModel<method, DataType>::Evaluate(
     const arma::mat& params, const DataType& data) {

     //Return a large error if parameters are out of bounds
     if ((params(0) > 1) || (params(0) < 0) ||
         (params(1) > 1) || (params(1) < 0) ||
         (params(2) > 1) || (params(2) < 0)) {

         return std::numeric_limits<double>::max();
     }

     //Getting the predictions from the given data
     arma::Row<double> predictions(data.n_elem);
     Predict<method, DataType>(
         data, predictions, params(0), params(1), params(2));

     //Computing and returning the squared error 
     //between the model predictions and the input data
     double error = 0;
     for (size_t i = m; i < data.n_elem; i++) {
         error += std::pow(data(i) - predictions(i), 2);
     }
     error /= (data.n_elem - m);
     return error;
 }


 template<char method, typename DataType>
 void  HoltWintersModel<method, DataType>::ResetData(
     DataType data, const size_t seasonal_period) {
     Data = std::move(data);
     m = seasonal_period;
 }
 template<char method, typename DataType>
 void  HoltWintersModel<method, DataType>::Predict(
     arma::Row<double>& predictions, size_t hMax) {
     Predict<method, DataType>(Data, predictions, 
         smoothingParams(0),
         smoothingParams(1),
         smoothingParams(2),
         hMax);
 }

 template<char method, typename DataType>
 void  HoltWintersModel<method, DataType>::Predict(
     const DataType& data, arma::Row<double>& predictions, size_t hMax) {
     Predict<method, DataType>(data, predictions,
         smoothingParams(0),
         smoothingParams(1),
         smoothingParams(2),
         hMax);
 }

template<char method, typename DataType>
template<char _method, typename _DataType, typename>
void  HoltWintersModel<method, DataType>::Predict(
    const DataType& data,
    arma::Row<double>& predictions,
    double alpha,
    double beta,
    double gamma,
    size_t H)
{
    //Constructing the vectors that will store the 
    //level, trend and seasonal esitimates
    arma::Row<double> L(data.n_elem);
    arma::Row<double> B(data.n_elem);
    arma::Row<double> S(data.n_elem + H);

    //Initializing the level and trend estimate vectors
    Initialize(data, L, B);

    //Initializing the seasonal estimate vector
    for (size_t i = 0; i < m; i++) {
        S(i) = data(i) - L(m - 1);
        predictions(i) = arma::datum::nan;
    }

    //Computing the level, trend, and seasonal estimates and 
    //forecasts at each time step i.
    for (size_t i = m; i < data.n_elem; i++) {
        L(i) = alpha * (data(i) - S(i - m)) + (1 - alpha) * (L(i - 1) + B(i - 1));
        B(i) = beta * (L(i) - L(i - 1)) + (1 - beta) * B(i - 1);
        S(i) = gamma * (data(i) - L(i - 1) - B(i - 1)) + (1 - gamma) * S(i - m);
        predictions(i) = L(i) + B(i) + S(i);
    }
    //Computing the forecasts 'H' time steps into the future.
    for (size_t i = data.n_elem; i < (data.n_elem + H); i++) {
        predictions(i) = L(data.n_elem - 1) + i * B(data.n_elem - 1) 
            + S(i - m*(std::floor((i-data.n_elem)/m)+1));
    }
}

template<char method, typename DataType>
template<char _method, typename _DataType, typename, typename>
void  HoltWintersModel<method, DataType>::Predict(
    const DataType& data,
    arma::Row<double>& predictions,
    double alpha,
    double beta,
    double gamma,
    size_t H)
{
    //Constructing the vectors that will store the 
    //level, trend and seasonal esitimates
    arma::Row<double> L(data.n_elem);
    arma::Row<double> B(data.n_elem);
    arma::Row<double> S(data.n_elem + H);

    //Initializing the level and trend estimate vectors
    Initialize(data, L, B);

    //Initializing the seasonal estimate vector
    for (size_t i = 0; i < m; i++) {
        S(i) = data(i)/L(m - 1);
        predictions(i) = arma::datum::nan;
    }

    //Computing the level, trend, and seasonal estimates 
    //and forecasts at each time step i.
    for (size_t i = m; i < data.n_elem; i++) {
        L(i) = alpha * (data(i)/ S(i - m)) + (1 - alpha) * (L(i - 1) + B(i - 1));
        B(i) = beta * (L(i) - L(i - 1)) + (1 - beta) * B(i - 1);
        S(i) = gamma * (data(i)/(L(i - 1) + B(i - 1))) + (1 - gamma) * S(i - m);
        predictions(i) = (L(i) + B(i))*S(i);
    }

    //Computing the forecasts 'H' time steps into the future.
    for (size_t i = data.n_elem; i < data.n_elem + H; i++) {
        predictions(i) = (L(data.n_elem-1) + i*B(data.n_elem-1))
            *S(i - m * (std::floor((i - data.n_elem) / m) + 1));
    }

}
template<char method, typename DataType>
void  HoltWintersModel<method, DataType>::Initialize(
    const DataType& data, arma::Row<double>& L, arma::Row<double>& B) {

    //Set the level and trend estimates to NaN till time step m
    for (size_t i = 0; i < m-1; i++) {
        L(i) = arma::datum::nan;
        B(i) = arma::datum::nan;
    }

    //Level estimate at time step m is initialized to the 
    //average of the first period (m) data.
    //Trend estimate at time step m is initialized to the 
    //average of of the slopes for each period in the first two periods.
    for (size_t i = 0; i < m; i++) {
        L(m - 1) += data(i);
        B(m - 1) += data(m + i) - data(i);
    }

    L(m-1) /= m;
    B(m-1) /= m*m;
}

} // namespace ts
} // namespace mlpack

#endif