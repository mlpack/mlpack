/**
 * @file sgd_impl.hpp
 * @author Ranjan Mondal
 *
 * Implementation of Parallel stochastic gradient descent.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_PARALLELSGD_SGDP_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_PARALLELSGD_SGDP_IMPL_HPP

#include <mlpack/methods/regularized_svd/regularized_svd_function.hpp>
// In case it hasn't been included yet.
#include "sgdp.hpp"
namespace mlpack {
namespace optimization {

template<typename DecomposableFunctionType>
ParallelSGD<DecomposableFunctionType>::ParallelSGD(DecomposableFunctionType& function,
                                   const double stepSize,
                                   const size_t maxIterations,
                                   const double tolerance,
                                   const bool shuffle) :   
    function(function),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance),
    shuffle(shuffle)

{ /* Nothing to do. */ }



//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double ParallelSGD<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  //get maximum number of threads that will be running. with is defined by OMP_NUM_THREADS
  size_t num_thread=omp_get_max_threads();


  //vector of iterate. length of tIterate is same as number of threads available.
  std::vector<arma::mat> tIterate; 

  //initializing each element of tIterate with initial iterate value 
  for(size_t i=0;i<num_thread;i++)
  {
    tIterate.push_back(iterate);
  }

  //sumIterate is taken track the  sum  all other computed iterate value from each thread. 
  arma::mat sumIterate(iterate.n_rows,iterate.n_cols);
  sumIterate.zeros();
  int th_num;
  
  #pragma omp parallel  private(th_num) 
  {
      th_num=omp_get_thread_num();
      SGD<DecomposableFunctionType> sgd(function,stepSize,maxIterations,tolerance,shuffle);
      sgd.Optimize(tIterate[th_num]);
  }
  
  //k is taken to count number of thread which give valid output i.e not inf or not nan 
  int k=0;
  for(size_t i=0;i<num_thread;i++)
  {

     overallObjective=0;
     for (size_t i = 0; i < numFunctions; ++i)
     {
       overallObjective += function.Evaluate(sumIterate,i);
     }

     if (!(std::isnan(overallObjective) || std::isinf(overallObjective)))
     {
       sumIterate+=tIterate[i];
       k++;
     }
     

  }
  
  if(k==0)
  {
    return(overallObjective);
  }

  sumIterate=sumIterate/k;
  iterate=sumIterate;
  overallObjective=0;

  for (size_t i = 0; i < numFunctions; ++i)
  {
    overallObjective += function.Evaluate(sumIterate,i);
  }

  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
