/**
 * @file cg.hpp
 * @author Ranjan Mondal
 *
 * conjugate gradient (CG).
 *      */
#ifndef __MLPACK_CORE_OPTIMIZERS_CG_CG_HPP
#define __MLPACK_CORE_OPTIMIZERS_CG_CG_HPP

#include <mlpack/core.hpp>
#include "cg.hpp"

namespace mlpack {
namespace optimization {


  template<typename CGFunctionType>
  class CG   
  {
      CG(CGFunctionType &function, const double stepSize,const size_t maxIterations, double tolerance);
      
      double Optimize(arma::mat &coordinates);
      CGFunctionType& Function() { return function; }
      
          
    private:
       double tolerance;
       double stepSize;
       size_t maxIterations;
       CGFunctionType &function; 

  };

}// namespace optimization
}// namespace mlpack

#endif



