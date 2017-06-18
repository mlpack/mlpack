/**
 * @file test_function.hpp
 * @author Ryan Curtin
 *
 * Very simple test function for CMAES (the same test used in sgd optimiser)
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_CMAES_TEST_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_CMAES_TEST_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {
namespace test {

//! Very, very simple test function which is the composite of three other
//! functions.  Its a derivative free optimization resulting in no use of gradient
//! like in stochastic gradient descent and instead uses search points taken randomly
//! of a normal distribution and moving the mean of the population towards the best 
//! fitness offsprings.
class cmaesTestFunction
{
 public:
  //! Nothing to do for the constructor.
  cmaesTestFunction() { }

  //! Return 3 (the number of functions) = the variable N in CMAES class for dimension
  size_t NumFunctions() const;

  //! Get the starting point = CMAES class xstart array of dimension given by NumFunctions()
  arma::mat GetInitialPoint() const;

  //! Get the intial standard devaition = CMAES class stddiv array of dimenison given by NumFunction()
  arma::mat GetInitialStdDev() const;

  //! Evaluate a function.
  double Evaluate( double const *coordinates) const;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
