/***
 * test_functions.h
 * 
 * A collection of functions to test optimizers (in this case, L-BFGS).  These
 * come from the following paper:
 *
 * "Testing Unconstrained Optimization Software"
 *  Jorge J. Moré, Burton S. Garbow, and Kenneth E. Hillstrom. 1981.
 *  ACM Trans. Math. Softw. 7, 1 (March 1981), 17-41.
 *  http://portal.acm.org/citation.cfm?id=355934.355936
 *
 * @author Ryan Curtin
 */

#ifndef __OPTIMIZATION_TEST_FUNCTIONS_H
#define __OPTIMIZATION_TEST_FUNCTIONS_H

#include <fastlib/fastlib.h>

// To fulfill the template policy class 'FunctionType', we must implement
// the following:
//
//   FunctionType(); // constructor
//   void Gradient(const arma::vec& coordinates, arma::vec& gradient);
//   double Evaluate(const arma::vec& coordinates);
//   const int GetDimension();
//   const arma::vec& GetInitialPoint();
//

// these names should probably be changed later
namespace mlpack {
namespace optimization {
namespace test {

/***
 * The Rosenbrock function, defined by
 *  f1(x) = 100 (x2 - x1^2)^2
 *  f2(x) = (1 - x1)^2
 *  x_0 = [-1.2, 1]
 *
 * This should optimize to f(x) = 0, at x = [1, 1].
 *
 * "An automatic method for finding the greatest or least value of a function."
 *   H.H. Rosenbrock.  1960.  Comput. J. 3., 175-184.
 */
class RosenbrockFunction {
 public:
  RosenbrockFunction(); // initialize initial point

  void Gradient(const arma::vec& coordinates, arma::vec& gradient);
  double Evaluate(const arma::vec& coordinates); 

  const int GetDimension() { return 2; }
  const arma::vec& GetInitialPoint();

 private:
  arma::vec initial_point;
};

}; // namespace test
}; // namespace optimization
}; // namespace mlpack

#endif
