/**
 * @file sgd_impl.hpp
 * @author Ranjan Mondal
 *
 * Implementation of stochastic gradient descent.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SGD_SGDP_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SGD_SGDP_IMPL_HPP

#include <mlpack/methods/regularized_svd/regularized_svd_function.hpp>
// In case it hasn't been included yet.
#include "sgd_P.hpp"
#include<iostream>
#include<omp.h>
namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
SGD_P<DecomposableFunctionType>::SGD_P(DecomposableFunctionType& function,
                                   const double stepSize,
                                   const size_t maxIterations,
                                   const double tolerance,
                                   const bool shuffle) :
    function(function),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }


template<typename DecomposableFunctionType>  arma::mat  SGD_P<DecomposableFunctionType>::Optimize_P(arma::mat iterate)
{
    const size_t numFunctions = function.NumFunctions();
    size_t   selectedFunction;
    double   objective = 0;
    arma::mat gradient(iterate.n_rows, iterate.n_cols);
    
    for(int i=0;i<maxIterations;i++)
    {
        selectedFunction=std::rand()%numFunctions;
        function.Gradient(iterate,selectedFunction, gradient);
        iterate -= stepSize * gradient;
        objective += function.Evaluate(iterate,selectedFunction);
    }
    return iterate;
}




//cordinate=irterate =variables
//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double SGD_P<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // This is used only if shuffle is true.
  // To keep track of where we are and how things are going.
  double overallObjective = 0;

  // Calculate the first objective function.
  for (size_t i = 0; i < numFunctions; ++i)
    overallObjective += function.Evaluate(iterate, i);

  arma::mat gradient(iterate.n_rows, iterate.n_cols);
 
  arma::mat iterate1(iterate.n_rows,iterate.n_cols);   
  iterate1.zeros();

  #pragma omp parallel  //   reduction(+:iterate1)
  {
//  for (size_t i = 1; i<=4;i++)
  
        iterate1+=Optimize_P(iterate);
        std::cout<<"done"<<std::endl; 
  }
 iterate1=iterate1/4; 

  overallObjective = 0;
  for (size_t i = 0; i < numFunctions; ++i)
  {
        overallObjective += function.Evaluate(iterate1, i);
  }
  iterate=iterate1;
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
