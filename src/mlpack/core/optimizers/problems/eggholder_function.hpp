/**
 * @file eggholder_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Eggholder function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_EGGHOLDER_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_EGGHOLDER_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
namespace test {

/**
 * The Eggholder function, defined by
 *
 * \f[
 * f(x) = -(x_2 + 47) * \sin(\sqrt(\left|x_2 + x_1 / 2 + 47\right|)) - x_1 *
 *    \sin(\sqrt(\left|x_1-(x_2 + 47)\right|))
 * \f]
 *
 * This should optimize to f(x) = -959.6407, at x = [512, 404.2319].
 *
 * For more information, please refer to:
 *
 * @code
 * @article{Jamil2013,
 *   title   = {A Literature Survey of Benchmark Functions For Global
 *              Optimization Problems},
 *   author  = {Momin Jamil and Xin{-}She Yang},
 *   journal = {CoRR},
 *   year    = {2013},
 *   url     = {http://arxiv.org/abs/1308.4008}
 * }
 * @endcode
 */
class EggholderFunction
{
 public:
  //! Initialize the EggholderFunction.
  EggholderFunction();

  /**
  * Shuffle the order of function visitation. This may be called by the
  * optimizer.
  */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("-333; -333"); }

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
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_PROBLEMS_EGGHOLDER_FUNCTION_HPP
