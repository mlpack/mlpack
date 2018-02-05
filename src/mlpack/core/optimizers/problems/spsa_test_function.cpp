/**
 * @file sgd_test_function.cpp
 * @author Ryan Curtin
 *
 * Implementation of very simple test function for stochastic gradient descent
 * (SGD).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "spsa_test_function.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

arma::vec SPSATestFunction::Evaluate(const arma::mat& coordinates,
                                     const int& size) const
{
  arma::vec ans = arma::zeros<arma::vec>(size);
  for (size_t i = 0; i < size; i++)
  switch ((int)coordinates[i])
  {
    case 0:
      ans[i] = -std::exp(-std::abs(coordinates[0]));

    case 1:
      ans[i] = std::pow(coordinates[1], 2);

    case 2:
      ans[i] = std::pow(coordinates[2], 4) + 3 * std::pow(coordinates[2], 2);

    default:
      ans[i] = 0;
  }

  return ans;
}
