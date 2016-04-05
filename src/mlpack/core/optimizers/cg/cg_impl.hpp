/**
 * @file cg_impl.hpp
 * @author Ranjan Mondal
 *
 * Implementation of nonlinear conjugate gradient .
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_CG_CG_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_CG_CG_IMPL_HPP

#include "cg.hpp"

namespace mlpack {
namespace optimization {

template<typename CGFunctionType>
NonlinearCG<CGFunctionType>::NonlinearCG(CGFunctionType& function,
                                   const size_t maxIterations,
                                   const double tolerance) :
    function(function),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do. */ }


/* Hestenes-Stiefel's formula is used for beta. Many others can be added latter with minor modificaiton. */ 

template<typename CGFunctionType>
double NonlinearCG<CGFunctionType>::compute_beta(arma::mat con_direction,arma::mat old_gradient,arma::mat new_gradient)
{

  double beta;
  arma::mat diff=new_gradient-old_gradient;
  beta=arma::det((new_gradient.t()*diff)/(con_direction.t()*diff));

  return(beta);

}


//! Optimize the function (minimize).
template<typename CGFunctionType>
double NonlinearCG<CGFunctionType>::Optimize(arma::mat& iterate)
{

 //@con_direction is to keep track to conjugate direction. 
  arma::mat con_direction,old_gradient,new_gradient;
  double stepSize;
  double beta;
  double newObjective = 0;
  double lastObjective = DBL_MAX;

  con_direction=arma::mat(iterate.n_rows, iterate.n_cols);
  old_gradient=arma::mat(iterate.n_rows, iterate.n_cols);
  new_gradient=arma::mat(iterate.n_rows, iterate.n_cols);

  //computing gradient and storing it into con_direction;
  function.gradient(iterate,con_direction);

  //initial value of the funtion. 
  newObjective=function.Evaluate(iterate);

  for(size_t i=1; i!=maxIterations ; i++)
  {
    

    //calculating step size. which is depenent upon 
    stepSize=function.ComputeStepSize(iterate,con_direction);
  
  
    function.gradient(iterate,old_gradient);

    //updating iterate
    iterate=iterate+stepSize*con_direction;
   
    //computing  gradient wih updated iterate. 
    function.gradient(iterate,new_gradient);

    beta=compute_beta(con_direction,old_gradient,new_gradient);

    //computing conjugate direction
    con_direction=-new_gradient+beta*con_direction;
   

    
    lastObjective = newObjective;
    newObjective=function.Evaluate(iterate);

    if(std::abs(lastObjective - newObjective) < tolerance || arma::norm(new_gradient,1)==0 )
    {
       Log::Info << "CG is  minimized  " <<"; " << "terminating optimization." << std::endl;
      return newObjective;
    }

    if(std::isnan(newObjective) || std::isinf(newObjective))
    {
      Log::Warn << "CG: converged to " << newObjective << "; terminating" << " with failure. please check the function" << std::endl;
      return newObjective;
    }

  }
  //theoretically  it will converge with in  iteration < length of iterate. i.e number of variable in iterate. But it may take long for numerical error in calculation. 
  Log::Info << "CG: maximum iterations (" << maxIterations << ") reached; "<< "terminating optimization." << std::endl;

  return newObjective;
}
  
} // namespace optimization
} // namespace mlpack

#endif
