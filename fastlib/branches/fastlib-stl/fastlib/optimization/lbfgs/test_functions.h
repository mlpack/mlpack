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
 *  f(x) = f1(x) + f2(x)
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

  double Evaluate(const arma::vec& coordinates); 
  void Gradient(const arma::vec& coordinates, arma::vec& gradient);

  const int GetDimension() { return 2; }
  const arma::vec& GetInitialPoint();

 private:
  arma::vec initial_point;
};

/***
 * The Wood function, defined by
 *  f(x) = f1(x) + f2(x) + f3(x) + f4(x) + f5(x) + f6(x)
 *  f1(x) = 100 (x2 - x1^2)^2
 *  f2(x) = (1 - x1)^2
 *  f3(x) = 90 (x4 - x3^2)^2
 *  f4(x) = (1 - x3)^2
 *  f5(x) = 10 (x2 + x4 - 2)^2
 *  f6(x) = (1 / 10) (x2 - x4)^2
 *  x_0 = [-3, -1, -3, -1]
 *
 * This should optimize to f(x) = 0, at x = [1, 1, 1, 1].
 *
 * "A comparative study of nonlinear programming codes."
 *   A.R. Colville.  1968.  Rep. 320-2949, IBM N.Y. Scientific Center.
 */
class WoodFunction {
 public:
  WoodFunction(); // initialize initial point

  double Evaluate(const arma::vec& coordinates);
  void Gradient(const arma::vec& coordinates, arma::vec& gradient);

  const int GetDimension() { return 4; }
  const arma::vec& GetInitialPoint();

 private:
  arma::vec initial_point;
};

/***
 * The Generalized Rosenbrock function in n dimensions, defined by
 *  f(x) = sum_i^{n - 1} (f(i)(x))
 *  f_i(x) = 100 * (x_i^2 - x_{i + 1})^2 + (1 - x_i)^2
 *  x_0 = [-1.2, 1, -1.2, 1, ...]
 *
 * This should optimize to f(x) = 0, at x = [1, 1, 1, 1, ...].
 *
 * "An analysis of the behavior of a glass of genetic adaptive systems."
 *   K.A. De Jong.  Ph.D. thesis, University of Michigan, 1975.
 */
class GeneralizedRosenbrockFunction {
 public:
  /***
   * Set the dimensionality of the extended Rosenbrock function.
   *
   * @param n Number of dimensions for the function.
   */
  GeneralizedRosenbrockFunction(int n);

  double Evaluate(const arma::vec& coordinates);
  void Gradient(const arma::vec& coordinates, arma::vec& gradient);

  const int GetDimension() { return n; }
  const arma::vec& GetInitialPoint();

 private:
  arma::vec initial_point;
  int n; // Dimensionality
};

}; // namespace test
}; // namespace optimization
}; // namespace mlpack

#endif
