/**
 * @file generalized_rosenbrock_function.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Definition of the Generalized Rosenbrock function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_GENERALIZED_ROSENBROCK_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_GENERALIZED_ROSENBROCK_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
namespace test {

/**
 * The Generalized Rosenbrock function in n dimensions, defined by
 *  f(x) = sum_i^{n - 1} (f(i)(x))
 *  f_i(x) = 100 * (x_i^2 - x_{i + 1})^2 + (1 - x_i)^2
 *  x_0 = [-1.2, 1, -1.2, 1, ...]
 *
 * This should optimize to f(x) = 0, at x = [1, 1, 1, 1, ...].
 *
 * This function can also be used for stochastic gradient descent (SGD) as a
 * decomposable function (DecomposableFunctionType), so there are other
 * overloads of Evaluate() and Gradient() implemented, as well as
 * NumFunctions().
 *
 * For more information, please refer to:
 *
 * @code
 * @phdthesis{Jong1975,
 *   title  = {Analysis of the behavior of a class of genetic adaptive
 *             systems},
 *   author = {De Jong, Kenneth Alan},
 *   school = {Queensland University of Technology},
 *   year   = {1975},
 *   type   = {{PhD} dissertation},
 * }
 * @endcode
 */
class GeneralizedRosenbrockFunction
{
 public:
  /*
   * Initialize the GeneralizedRosenbrockFunction.
   *
   * @param n Number of dimensions for the function.
   */
  GeneralizedRosenbrockFunction(const size_t n);

  /**
  * Shuffle the order of function visitation. This may be called by the
  * optimizer.
  */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return n - 1; }

  //! Get the starting point.
  const arma::mat& GetInitialPoint() const { return initialPoint;}

  /*
   * Evaluate a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param batchSize Number of points to process.
   */
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize = 1) const;

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
                const size_t batchSize = 1) const;

  /*
   * Evaluate the gradient of a function for a particular batch-size.
   *
   * @param coordinates The function coordinates.
   * @param begin The first function.
   * @param gradient The function gradient.
   */
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::sp_mat& gradient,
                const size_t count) const;

  /*
   * Evaluate the gradient of a function with the given coordinates.
   *
   * @param coordinates The function coordinates.
   * @param gradient The function gradient.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const;

 private:
  //! Locally-stored Initial point.
  arma::mat initialPoint;

  //! //! Number of dimensions for the function.
  size_t n;

  //! For shuffling.
  arma::Row<size_t> visitationOrder;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_PROBLEMS_GENERALIZED_ROSENBROCK_FUNCTION_HPP
