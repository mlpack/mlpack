/**
 * @file rosenbrock_function.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Definition of the Rosenbrock function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_ROSENBROCK_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_ROSENBROCK_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
namespace test {

/**
 * The Rosenbrock function, defined by:
 *
 *  f(x) = f1(x) + f2(x)
 *  f1(x) = 100 (x2 - x1^2)^2
 *  f2(x) = (1 - x1)^2
 *  x_0 = [-1.2, 1]
 *
 * This should optimize to f(x) = 0, at x = [1, 1].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Rosenbrock1960,
 *    title  = {An Automatic Method for Finding the Greatest or Least Value of a
 *              Function},
 *   author  = {Rosenbrock, H. H.},
 *   journal = {The Computer Journal},
 *   number  = {3},
 *   pages   = {175--184},
 *   year    = {1960},
 * }
 * @endcode
 */
class RosenbrockFunction
{
 public:
  //! Initialize the RosenbrockFunction.
  RosenbrockFunction();

  /**
  * Shuffle the order of function visitation. This may be called by the
  * optimizer.
  */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("-1.2; 1"); }

  /*
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize) const;

  /*
   * Evaluate a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   */
  double Evaluate(const arma::mat& coordinates) const;

  /*
   * Evaluate the gradient of a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   * @param batchSize Number of points to process.
   */
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize) const;

  /*
   * Evaluate the gradient of a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   * @param gradient The function gradient.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_PROBLEMS_ROSENBROCK_FUNCTION_HPP
