/**
 * @file scd.hpp
 * @author Ranjan Mondal
 *
 * Stochastic Coordinate Descent (SGD).
 *      */
#ifndef __MLPACK_CORE_OPTIMIZERS_SCD_SCD_HPP
#define __MLPACK_CORE_OPTIMIZERS_SCD_SCD_HPP

#include <mlpack/core.hpp>
#include "scd.hpp"

namespace mlpack {
namespace optimization {


  template<typename SCDFunctionType>
  class SCD   
  {
      SCD(SCDFunctionType &function, const double stepSize,const size_t maxIterations, double tolerance);
      
      double Optimize(arma::mat &coordinates);
      SCDFunctionType& Function() { return function; }
      
          
    private:
       double tolerance;
       double stepSize;
       size_t maxIterations;
       SCDFunctionType &function; 

  };

}// namespace optimization
}// namespace mlpack

#endif



