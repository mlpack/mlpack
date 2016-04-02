/**
 * @file test_function.hpp
 * @author Ranjan Mondal
 *
 * Very simple test function for SCD.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SCD_TEST_FUNCTION_HPP
#define __MLPACK_CORE_OPTIMIZERS_SCD_TEST_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {
namespace test {

//! Very, very simple test function which is the composite of three other
//! functions.  The gradient is not very steep far away from the optimum, so a
//! larger step size may be required to optimize it in a reasonable number of
//! iterations.
class SCDTestFunction
{
 public:
  //! Nothing to do for the constructor.
  SGDTestFunction() { }

  //! Return 3 (the number of functions).
  size_t NumCoordinates() const { return 3; }

  //! Get the starting point.
  arma::mat GetInitialCoordinats() const {return arma::mat("6; -45.6; 6.2");} 

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates, const size_t i) const;

  //returns ith Lipschitz Constant 
  double Lipschitz_Constant(const arma::mat& coordinates, const size_t i);  const; 


 //returns L_max. it is dependent upon the objective funtion we are  maximizing,
  //double Lipschitz_Constant_max(void);  const;  //this may be usd in AsySCD

  //Evaluates the gradient along  ith coordinate at current cordinate value and updates gradient at ith place. 
  void coordinate_gradient(arma::mat& coordinates,const size_t i,arma::mat& gradient);


};


} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
