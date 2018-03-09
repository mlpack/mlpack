/**
 * @file rosenbrock_wood_function.hpp
 * @author Ryan Curtin
 * @author Marcus Edel
 *
 * Definition of the Rosenbrock-Wood function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>
#include <mlpack/core/optimizers/problems/wood_function.hpp>

namespace mlpack {
namespace optimization {
namespace test {

/**
 * The Generalized Rosenbrock function in 4 dimensions with the Wood Function in
 * four dimensions.  In this function we are actually optimizing a 2x4 matrix of
 * coordinates, not a vector.
 */
class RosenbrockWoodFunction
{
 public:
  //! Initialize the RosenbrockWoodFunction.
  RosenbrockWoodFunction();

  /**
  * Shuffle the order of function visitation. This may be called by the
  * optimizer.
  */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

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

 private:
  //! Locally-stored initial point.
  arma::mat initialPoint;

  //! Locally-stored Generalized-Rosenbrock function.
  GeneralizedRosenbrockFunction rf;

  //! Locally-stored Wood function.
  WoodFunction wf;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_PROBLEMS_ROSENBROCK_WOOD_FUNCTION_HPP
