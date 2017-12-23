/**
 * @file rastrigin_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Rastrigin function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_RASTRIGIN_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_RASTRIGIN_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
namespace test {

/**
 * The Rastrigin function, defined by
 *
 * \f[
 * f(x) = 10 * d * \sum_{i=1}^{d} x_i^2 - 10 * \cos(2 * \pi * x_i)
 * \f]
 *
 * This should optimize to f(x) = 0
 * at x = [0, ..., 0].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Jamil2013,
 *   title   = {Systems of extremal control},
 *   author  = {Rastrigin, L. A.},
 *   journal = {Mir},
 *   year    = {1974}
 * }
 * @endcode
 */
class RastriginFunction
{
 public:
  /*
   * Initialize the RastriginFunction.
   *
   * @param n Number of dimensions for the function.
   */
  RastriginFunction(const size_t n);

  /**
  * Shuffle the order of function visitation. This may be called by the
  * optimizer.
  */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return n; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return initialPoint; }

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
  void Gradient(const arma::mat& coordinates, arma::mat& gradient);
 private:
  //! Number of dimensions for the function.
  size_t n;

  //! For shuffling.
  arma::Row<size_t> visitationOrder;

  //! Initial starting point.
  arma::mat initialPoint;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_PROBLEMS_RASTRIGIN_FUNCTION_HPP
