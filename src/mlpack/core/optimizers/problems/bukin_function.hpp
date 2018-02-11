/**
 * @file bukin_function.hpp
 * @author Marcus Edel
 *
 * Definition of the Booth function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_BUKIN_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_BUKIN_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
namespace test {

/**
 * The Bukin function, defined by
 *
 * \f[
 * f(x) = 100 * \sqrt(\left|x_2 - 0.01 * x_1^2 \right|) +
 *    0.01 * \left|x_1 + 10 \right|
 * \f]
 *
 * This should optimize to f(x) = 0, at x = [-10, 1].
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
class BukinFunction
{
 public:
  /*
   * Initialize the BukinFunction.
   *
   * @param epsilon Coefficient to avoid division by zero (numerical stability).
   */
  BukinFunction(const double epsilon = 1e-8);

  /**
  * Shuffle the order of function visitation. This may be called by the
  * optimizer.
  */
  void Shuffle();

  //! Return 1 (the number of functions).
  size_t NumFunctions() const { return 1; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("-10; -2.0"); }

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

  //! Get the value used for numerical stability.
  double Epsilon() const { return epsilon; }
  //! Modify the value used for numerical stability.
  double& Epsilon() { return epsilon; }

 private:
  //! The value used for numerical stability.
  double epsilon;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif // MLPACK_CORE_OPTIMIZERS_PROBLEMS_BUKIN_FUNCTION_HPP
