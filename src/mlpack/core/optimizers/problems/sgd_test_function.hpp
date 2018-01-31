/**
 * @file sgd_test_function.hpp
 * @author Ryan Curtin
 *
 * Very simple test function for SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROBLEMS_SGD_TEST_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PROBLEMS_SGD_TEST_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
namespace test {

//! Very, very simple test function which is the composite of three other
//! functions.  The gradient is not very steep far away from the optimum, so a
//! larger step size may be required to optimize it in a reasonable number of
//! iterations.
class SGDTestFunction
{
 private:
  arma::Col<size_t> visitationOrder;

 public:
  //! Initialize the SGDTestFunction.
  SGDTestFunction();

  /**
  * Shuffle the order of function visitation.  This may be called by the optimizer.
  */
  void Shuffle();

  //! Return 3 (the number of functions).
  size_t NumFunctions() const { return 3; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("6; -45.6; 6.2"); }

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates, const size_t i) const;

  //! Evaluate a function for a particular batch-size.
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize) const;

  //! Evaluate the gradient of a function.
  void Gradient(const arma::mat& coordinates,
                const size_t i,
                arma::mat& gradient) const;

  //! Evaluate the gradient of a function for a particular batch-size
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize) const;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
