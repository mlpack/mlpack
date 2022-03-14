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
#include<ensmallen.hpp>

namespace mlpack {
namespace ts {
template<typename datatype = arma::Row<double>>
class SimpleExpo
{

SimpleExpo();


SimpleExpo(datatype data,double alpha,double level);


SimpleExpo(datatype data);


double Loss();

double Loss(const arma::mat & params);

void Train();

void Predict(const datatype &data,arma::Row<double>& predictions,size_t forcast);
void Predict(arma::Row<double>& predictions,size_t forcast);

double const &alpha()
{
  return params[0];
}

double const &level()
{
  return params[1];
}

double & alpha();
double & level();
void Initialize();

private:

template<typename Datatype ,typename = std::enable_if_t<((arma::is_Row<Datatype>::value||arma::is_Col<Datatype>::value))>>
void Predict(const datatype & data,arma::Row<double> & predictions,double alpha ,double level,size_t forcast)

template<typename Datatype,typename=std::enable_if_t<((arma::is_Row<Datatype>::value || arma::is_Col<Datatype>::value))>,typename = void>
void Predict(const Datatype & data,arma::Row<double>& predictions,double alpha,double level,size_t forcast = 0);

size_t forcast;

datatype data;

arma::mat params;

double Loss(const arma::mat & params,const datatype & data);

};
}
}
#endif