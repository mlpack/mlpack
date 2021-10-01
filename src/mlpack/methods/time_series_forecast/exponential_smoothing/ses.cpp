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

#include <mlpack/core/util/log.hpp>
#include "ses.hpp"
#include <cmath>
#include <numeric>

using namespace mlpack;
using namespace mlpack::ts;
// using namespace mlpack::util;
using namespace arma;
using namespace std;

// Calculate errors

double CalcMae(arma::rowvec& errors) {
    double mae = 0.0;
    for(arma::uword c=0; c < errors.n_elem; ++c){
	mae += std::abs(errors(c));
    }
    return mae/errors.n_elem;
}

double CalcMse(arma::rowvec& errors) {
    double mse = 0.0;
    for(arma::uword c=0; c < errors.n_elem; ++c){
	mse += pow(errors(c), 2);
    }
    return mse/errors.n_elem;
}

double CalcMape(arma::rowvec& data, arma::rowvec& errors) {
    double mape = 0.0;
    for(arma::uword c1=0, c2=0; c1 < errors.n_elem, c2 < data.n_elem; ++c1, ++c2){
	    mape += std::abs(errors(c1) / data(c2));
    }
    return mape/errors.n_elem;
}

//Function declaration for the SES 

SingleES::SingleES(arma::rowvec& data, const double& alpha) {
    this->alpha = alpha;
    int sZ =0;
    // forecastPast.push_back(data.front());
    forecastPast = {data.front()};
    
    //  errorForecastPast.push_back(0);
    errorForecastPast = {0};

    // Evaluate the output with the help of smoothing factor
    
    for(arma::uword c=0; c < data.n_elem-1; ++c){
    // for(std::vector<double>::iterator it = data.begin(); it != data.end()-1; it++) {
        // forecastPast.push_back(alpha*(*it) + (1-alpha)*(forecastPast.back()));
        
        sZ = forecastPast.size();
    	forecastPast.resize(sZ+1);
    	forecastPast(sZ) = alpha*(data(c)) + (1-alpha)*(forecastPast.back());
        
        // errorForecastPast.push_back(forecastPast.back() - (*(it+1)));
        
        sZ = errorForecastPast.size();
    	errorForecastPast.resize(sZ+1);
    	errorForecastPast(sZ) = forecastPast.back() - (data(c+1));
        
    }
    // forecastPast.push_back(alpha*data.back() + (1-alpha)*forecastPast.back());
    sZ = forecastPast.size();
    forecastPast.resize(sZ+1);
    forecastPast(sZ) = alpha*data.back() + (1-alpha)*forecastPast.back();
}

// output for the model 

void SingleES::SesForecastSummary(arma::rowvec& data, SingleES& sesObject) {
    cout<<"Forecast with Alpha="<<sesObject.Getalpha()<<"\n";
    int future = 1;
    arma::rowvec &foreVec = sesObject.GetForecastVector(), &errVec = sesObject.GetErrorVector();
    for(arma::uword c1=0, c2=0; c1 < foreVec.n_elem, c2 < errVec.n_elem; ++c1, ++c2){
        cout<<"   "<<future<<"->"<<foreVec(c1)<<"   "<<"E->"<<errVec(c2)<<"\n";
        future++;
    }
    cout<<"Forecast for all future periods\n   "<<foreVec.back()<<"\n";
    cout<<"\nForecast Evaluation\n";
    cout<<"   Mean absolute error :"<<CalcMae(errVec)<<"\n";
    cout<<"   Mean squared error :"<<CalcMse(errVec)<<"\n";
    cout<<"   Mean absolute percentage error :"<<CalcMape(data, errVec)<<"\n";
}

//Get the private variables with functions

arma::rowvec& SingleES::GetForecastVector() {
    return forecastPast;
}

arma::rowvec& SingleES::GetErrorVector() {
    return errorForecastPast;
}

double SingleES::Getalpha() const {
    return alpha;
}
