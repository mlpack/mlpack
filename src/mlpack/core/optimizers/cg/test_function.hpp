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

/*
 * This class represents the function 4*x^2+8*y^2-4*x-8*y+15
 * same as matrix form (1/2)[x y] [16 0][x]   +[x y][-4]  + 15
 *                                [0  8][y]         [-8]
 * we can represent this function as (1/2)X'AX + X'B +15  ;
 *  where X'=[x y] ;A=[[16,0],[0,8]] B=[-4,-8] in row major ; 
 * gradient = AX + B
 * minimum value=12   at X=[x y]=[0.5 0.5]
 * 
 */
  
class CGTestFunction1
{
 public:
  //! Nothing to do for the constructor.
  CGTestFunction1() {
    A=arma::mat("8,0;0,16");
    B=arma::mat("-4;-8");
    c=15;
  }


  //! Get the starting point.
  arma::mat GetInitialPoint() const; // {return arma::mat(" 5;6");} 

  //! Evaluate a function.
  double Evaluate(const arma::mat& iterate) const;

  void gradient(arma::mat& iterate,arma::mat& gradient);
  
  //it computes  minimizing the function f(x_i+ step_size*gradient_direction) 
  double ComputeStepSize(arma::mat& iterate,arma::mat& gradient);
 
   private:
    arma::mat A;
    
    arma::mat B;
    double c;

};






/*
 * This class represents the function 4*x^2+8*y^2-4*x-8*y+13
 * same as matrix form (1/2)[x y][2 0][x]   +[x y][-4]  + 13
 *                               [0 2][y]         [-6]
 * we can represent this function as (1/2)X'AX + X'B +15  ;
 *  where X'=[x y] ;A=[[16,0],[0,8]] B=[-4,-8] in row major ; 
 * gradient = AX + B
 * minimum value=0   at X=[x y]=[2 3]
 * 
 */
class CGTestFunction2
{
 public:
  //! Nothing to do for the constructor.
  CGTestFunction2() {

      A=arma::mat("2,0;0,2");
      B=arma::mat("-4;-6");
      c=13; 

  }


  //! Get the starting point.
  arma::mat GetInitialPoint() const; // {return arma::mat(" 5;6");} 

  //! Evaluate a function.
  double Evaluate(const arma::mat& iterate) const;

  void gradient(arma::mat& iterate,arma::mat& gradient);
  
  //it computes  minimizing the function  f(x_i+ step_size*gradient_direction) with respect to step_size. varient line search method can be applied here. 
  double ComputeStepSize(arma::mat& iterate,arma::mat& gradient);
 
   private:
    arma::mat A;
    arma::mat B;
    double c;


};


} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
