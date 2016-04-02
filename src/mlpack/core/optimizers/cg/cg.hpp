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
  class NonlinearCG 
  {
    public:
      NonlinearCG(CGFunctionType &function, const size_t maxIterations=10000, const double tolerance=1e-5);
      CGFunctionType& Function() { return function; }
      
      double Optimize(arma::mat &iterate);
      double compute_beta(arma::mat con_direction,arma::mat old_gradient,arma::mat new_gradient);
 
      double Tolerance() const { return tolerance; }
   
      size_t MaxIterations() const { return maxIterations; }
   
      const CGFunctionType& Function() const { return function; }
      
     private:
       CGFunctionType &function; 
      
       //maximum number of iteration. 
       const size_t maxIterations; 

       //The tolerance for termination.
       const double tolerance;
  };

}// namespace optimization
}// namespace mlpack

#include"cg_impl.hpp"

#endif



