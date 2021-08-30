/**
 * @file methods/time_series/exponential_smoothing/ses.cpp
 * @author Aditi Pandey
 *
 * Implementation of Single exponential smoothing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "ses.hpp"
#include <mlpack/core/util/log.hpp>

using namespace mlpack;
using namespace mlpack::single_exponential_smoothing;


double calc_mae(std::vector<double>& errors) {
    double mae = 0.0;
    for(std::vector<double>::iterator it = errors.begin(); it != errors.end(); it++) {
        mae += std::abs(*it);
    }
    return mae/errors.size();
}

double calc_mse(std::vector<double>& errors) {
    double mse = 0.0;
    for(std::vector<double>::iterator it = errors.begin(); it != errors.end(); it++) {
        mse += pow(*it, 2);
    }
    return mse/errors.size();
}

double calc_mape(std::vector<double>& data, std::vector<double>& errors) {
    double mape = 0.0;
    for(std::vector<double>::iterator it1 = errors.begin(), it2 = data.begin();\
    it1 != errors.end(), it2 != data.end(); it1++, it2++) {
        mape += std::abs(*it1 / *it2);
    }
    return mape/errors.size();
}

void SingleES::update(const double& data, const double& alpha) {
    forecast_past.push_back(alpha*data + (1-alpha)*forecast_past.back());
}

void SingleES::SingleES(std::vector<double>& data, const double& alpha) {
    this->alpha = alpha;
    forecast_past.push_back(data.front());
    error_forecast_past.push_back(0);
    for(std::vector<double>::iterator it = data.begin(); it != data.end()-1; it++) {
        forecast_past.push_back(alpha*(*it) + (1-alpha)*(forecast_past.back()));
        error_forecast_past.push_back(forecast_past.back() - (*(it+1)));
    }
    forecast_past.push_back(alpha*data.back() + (1-alpha)*forecast_past.back());
}

// output for the model 

void SingleES::ses_forecast_summary(QString& s, std::vector<double>& d, SingleES& h) {
    std::stringstream out;
    out<<"Forecast with Alpha="<<h.get_alpha()<<"\n";
    int future = 1;
    std::vector<double> &fvec = h.forecast_vector_ref(), &errvec = h.error_vector_ref();
    for(std::vector<double>::iterator i1=fvec.begin(), i2=errvec.begin();\
        i1 != fvec.end(), i2 != errvec.end(); i1++, i2++) {
        out<<"   "<<future<<"->"<<*i1<<"   "<<"E->"<<*i2<<"\n";
        future++;
    }
    out<<"Forecast for all future periods\n   "<<fvec.back()<<"\n";
    out<<"\nForecast Evaluation\n";
    out<<"   Mean absolute error :"<<calc_mae(errvec)<<"\n";
    out<<"   Mean squared error :"<<calc_mse(errvec)<<"\n";
    out<<"   Mean absolute percentage error :"<<calc_mape(d, errvec)<<"\n";
    s += out.str().c_str();
}


std::vector<double>& SingleES::forecast_vector_ref() {
    return forecast_past;
}

std::vector<double>& SingleES::error_vector_ref() {
    return error_forecast_past;
}

double SingleES::get_alpha() const {
    return _alpha;
}
