/**
 * @file sgd_test_function_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of very simple test function for stochastic gradient descent
 * (SGD).
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SGD_TEST_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SGD_TEST_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "sgd_test_function.hpp"

namespace ens {
namespace test {

inline SGDTestFunction::SGDTestFunction() :
    visitationOrder(arma::linspace<arma::Col<size_t>>(0, NumFunctions() - 1,
        NumFunctions()))
{ }

inline void SGDTestFunction::Shuffle()
{
  visitationOrder = arma::shuffle(arma::linspace<arma::Col<size_t> >(0,
      (NumFunctions() - 1), NumFunctions()));
}

inline double SGDTestFunction::Evaluate(const arma::mat& coordinates,
                                        const size_t begin,
                                        const size_t batchSize) const
{
  double objective = 0;

  for (size_t i = begin; i < begin + batchSize; i++)
  {
    switch (visitationOrder(i))
    {
      case 0:
        objective -= std::exp(-std::abs(coordinates[0]));
        break;

      case 1:
        objective += std::pow(coordinates[1], 2);
        break;

      case 2:
        objective += std::pow(coordinates[2], 4) + \
                     3 * std::pow(coordinates[2], 2);
        break;
    }
  }

  return objective;
}

inline void SGDTestFunction::Gradient(const arma::mat& coordinates,
                                      const size_t begin,
                                      arma::mat& gradient,
                                      const size_t batchSize) const
{
  gradient.zeros(3);

  for (size_t i = begin; i < begin + batchSize; ++i)
  {
    switch (visitationOrder(i))
    {
      case 0:
        if (coordinates[0] >= 0)
          gradient[0] += std::exp(-coordinates[0]);
        else
          gradient[0] += -std::exp(coordinates[0]);
        break;

      case 1:
        gradient[1] += 2 * coordinates[1];
        break;

      case 2:
        gradient[2] += 4 * std::pow(coordinates[2], 3) + 6 * coordinates[2];
        break;
    }
  }
}

} // namespace test
} // namespace ens

#endif
