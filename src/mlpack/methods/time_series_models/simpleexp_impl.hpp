/**
 * @file methods/time_series_models/simpleexpo_impl.hpp
 * @author Rishabh Bali
 *
 * Implementation of the SimpleExpo Model for time series data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TIME_SERIES_MODELS_SIMPLEEXPO_MODEL_IMPL_HPP
#define MLPACK_METHODS_TIME_SERIES_MODELS_SIMPLEEXPO_MODEL_IMPL_HPP

#include <mlpack/prereqs.hpp>
#include<mlpack/core/cv/metrics/mse.hpp>

// In case it hasn't been included already

#include "simpleexpo.hpp"

namespace mlpack
{
    namespace ts
    {
   template< typename datatype>
   SimpleExpo<datatype>::SimpleExpo()
   {
       Initialize();
   }
     template<typename datatype>
     SimpleExpo<datatype>::SimpleExpo(datatype data) : data(data)
     {
         Initialize();
     }
     template<typename datatype>
     SimpleExpo<datatype>::SimpleExpo(datatype data,double alpha,double level):data(data),params({aplha,level})
     {  }
    

     template<typename datatype>
     void SimpleExpo<datatype>::Train()
     {
         ens::GradientDescent optimizer;
         optimizer.Optimize(*this,params);
     }
     template<typename datatype>
     void SimpleExpo<datatype>::Initialize()
     {
        double alpha = ((double)rand()/(double)RAND_MAX);
        double level = data.row(data.n_rows-1)(0);
        params[{alpha,level}];
     }
     template<typename datatype>
     double SimpleExpo<datatype>::Loss(const arma::mat & params)
     {
         return Loss(params,data);
     }
     template<typename datatype>
     double SimpleExpo<datatype>::Loss()
     {
         return Loss(params,data);
     }

     template<typename datatype>
     double SimpleExpo<datatype>::Loss(const arma::mat& params,const datatype & data)
     {
         if(params(0)>1 || params(0)<0)
         {
             return std::numeric_limits<double>::max();
         }
         arma::Row<double> predictions(data.n_elem);
         Predict<dataype>(data,predictions,params(0),params(1));
         double error = 0;
         for(size_t i =0;i<data.n_elem;i++)
         {
             error += std::pow(data(i)-predictions(i),2);
         }
         return error;
     }

     template<typename datatype>
     void SimpleExpo<datatype>::Predict(const datatype& data,arma::Row<double> &predictions,size_t forcast)
     {
         return Predict<datatype>(data,predictions,params(0),params(1),forcast);
     }
     template<typename datatype>
     void SimpleExpo<datatype>::Predict(arma::Row<double>& predictions,size_t forcast)
     {
         return Predict<dataype>(data,predictions,params(0),params(1),size_t forcast);
     }

     template<typename datatype>
     template<typename Datatype,typename>
     void SimpleExpo<datatype>::Predict(const datatype & data,arma::Row<double> & predictions,double alpha,double level,size_t forcast)
     {   prediction(0) = level;
        for(size_t i =1;i<data.n_elems;i++)
        {
            predictions(i) =  alpha * data(i-1) + aplha*(1-alpha)*predictions(i-1);
        }

        for(size_t j = 0;j<forcast;j++)
        {
            predictions(i+j) = alpha * prediction(i+j-1) + (alpha*(1-aplha))prediction(i+j-1)
        }
     }


}
}
#endif