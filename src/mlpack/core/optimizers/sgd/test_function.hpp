/**
 * @file test_function.hpp
 * @author Ryan Curtin
 *
 * Very simple test function for SGD.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_TEST_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_TEST_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {
namespace test {

//! Very, very simple test function which is the composite of three other
//! functions.  The gradient is not very steep far away from the optimum, so a
//! larger step size may be required to optimize it in a reasonable number of
//! iterations.
class SGDTestFunction
{
 public:
  //! Nothing to do for the constructor.
  SGDTestFunction() { }

  //! Return 3 (the number of functions).
  size_t NumFunctions() const { return 3; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("6; -45.6; 6.2"); }

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates, const size_t i) const;

  //! Evaluate the gradient of a function.
  void Gradient(const arma::mat& coordinates,
                const size_t i,
                arma::mat& gradient) const;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
