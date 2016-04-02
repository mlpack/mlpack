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
                                   const double tolerance) :   
    function(function),
    stepSize(stepSize),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }



//! Optimize the function (minimize).
template<typename DecomposableFunctionType>
double ParallelSGD<DecomposableFunctionType>::Optimize(arma::mat& iterate)
{

  // Find the number of functions to use.
  const size_t numFunctions = function.NumFunctions();

  // To keep track of where we are and how things are going.
  double overallObjective = 0;
  double lastObjective = DBL_MAX;

  //get maximum number of threads that will be running. with is defined by OMP_NUM_THREADS
  int num_thread=omp_get_max_threads();


  //vector of iterate. length of tIterate is same as number of threads available.
  std::vector<arma::mat> tIterate; 

  //initializing each element of tIterate with initial iterate value 
  for(int i=0;i<num_thread;i++)
  {
    tIterate.push_back(iterate);
  }

  //sumIterate is taken track the  sum  all other computed iterate value from each thread. 
  arma::mat sumIterate(iterate.n_rows,iterate.n_cols);

  int it;   
  bool halt=false;
  #pragma omp parallel  shared(sumIterate,halt) private(it) 
  {
    it=0; 
    while(it!=maxIterations && halt != true)
    {
      it++;
      int th_num=omp_get_thread_num(); //thread number is stored in which the thread is running. 
      arma::mat gradient(iterate.n_rows, iterate.n_cols);  //To make gradient private to each thread it is declared here.
      int selectedFunction;
    
      for(int j=0; j < numFunctions; j++)
      {
        selectedFunction=std::rand()%numFunctions;
        function.Gradient(tIterate[th_num],selectedFunction, gradient);
        tIterate[th_num] -= stepSize * gradient;
      }
      #pragma omp barrier    //wait untill all the threads reach here
   
      //this portion will run only by one thread
      #pragma omp master
      {
         sumIterate.zeros();
      }    
      #pragma omp barrier   

      //Taking the sum of all other computed iterate. Siterate is shared by all threads
      #pragma omp critical 
      {
        sumIterate += tIterate[th_num];
      }
    
      #pragma omp barrier   
      //runing  a single thread 
      #pragma omp master
      {
        sumIterate=sumIterate/num_thread;
        overallObjective=0;
        for (size_t i = 0; i < numFunctions; ++i)
        {
          overallObjective += function.Evaluate(sumIterate,i);
        }
/*
        if(it%1000==0 || it<100)
        {
        std::cout<<it<<"   "<<lastObjective<<"  "<<overallObjective<<std::endl;
        }
*/
        if (std::isnan(overallObjective) || std::isinf(overallObjective))
        {
          Log::Warn << "SGD: converged to " <<overallObjective << "; terminating"<< " with failure.  Try a smaller step size?" << std::endl;
          //halt=true; 
        }


        if (std::abs(lastObjective - overallObjective) < tolerance)
        {
          Log::Info << "SGD: minimized within tolerance " << tolerance << "; "<< "terminating optimization." << std::endl;
        //std::cout<<it<<"*   "<<lastObjective<<"  "<<overallObjective<<std::endl;
        //  halt=true; 
        }
        lastObjective=overallObjective;
      }
      #pragma omp barrier

    }   //end of while loop
  }   //end of all thread
  
  iterate=sumIterate;
  overallObjective=0;


  // Calculating the  objective function with computed iterate
  for (size_t i = 0; i < numFunctions; ++i)
  overallObjective += function.Evaluate(iterate, i);
  
  return overallObjective;
}

} // namespace optimization
} // namespace mlpack

#endif
