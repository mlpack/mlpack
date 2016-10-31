/**
 * @file test_function.hpp
 * @author Sumedh Ghaisas
 *
 * Very simple test function for SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_GD_TEST_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_GD_TEST_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {
namespace test {

//! Very, very simple test function which is the composite of three other
//! functions.  The gradient is not very steep far away from the optimum, so a
//! larger step size may be required to optimize it in a reasonable number of
//! iterations.
class GDTestFunction
{
 public:
  //! Nothing to do for the constructor.
  GDTestFunction() { }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("1; 3; 2"); }

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates) const;

  //! Evaluate the gradient of a function.
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
