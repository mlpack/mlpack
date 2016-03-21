/**
 * @file scd_impl.hpp
 * @author Ranjan Mondal
 *
 * Stochastic Coordinate Descent (SGD)
 *      */
#ifndef __MLPACK_CORE_OPTIMIZERS_SCD_SCDIMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SCD_SCDIMPL_HPP

#include <mlpack/core.hpp>
#include"scd.hpp"
namespace mlpack {
namespace optimization {


template<typename SCDFunctionType>
SCD<SCDFunctionType>::SCD(SCDFunctionType &function,const double stepSize, const size_t maxIterations, double tolerance):
  function(function),
  stepSize(stepSize),
  maxIterations(maxIterations),
  tolerance(tolerance)
  {     }
      
template<typename SCDFunctionType>
double SCD<SCDFunctionType>::Optimize(arma::mat &coordinates)
{

  double overallObjective = 0;
  double lastObjective = DBL_MAX;
  double lConstant;
  const size_t numCoordinate=function.NumCoordinate();
  int i;
  
  arma::mat CGradient(1,numCoordinate);  
  overallObjective=function.Evaluate(coordinates);    
  
  for (size_t j = 1; j <= maxIterations; j++)
  {
    i=std::rand()%numCoordinate;
    lConstant=function.Lipschitz_Constant(coordinates,i);
    function.coordinate_gradient(coordinates,i,CGradient);

   //ith coordinate update
    coordinates[i]=coordinates[i]- stepSize*(CGradient[i]/lConstant);
 

  //tollerence checking//

  }
  


}
      
          

}// namespace optimization
}// namespace mlpack

#endif



