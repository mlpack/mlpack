/**
 * @file sparse_test_function_impl.hpp
 * @author Shikhar Bhardwaj
 *
 * Sparse test function for Parallel SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_IMPL_HPP
#define ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "sparse_test_function.hpp"

namespace ens {
namespace test {

inline SparseTestFunction::SparseTestFunction()
{
  intercepts = arma::vec("20 12 15 100");
  bi = arma::vec("-4 -2 -3 -8");
}

//! Evaluate a function.
inline double SparseTestFunction::Evaluate(
    const arma::mat& coordinates,
    const size_t i,
    const size_t batchSize) const
{
  double result = 0.0;
  for (size_t j = i; j < i + batchSize; ++j)
  {
    result += coordinates[j] * coordinates[j] + bi[j] * coordinates[j] +
        intercepts[j];
  }

  return result;
}

//! Evaluate all the functions.
inline double SparseTestFunction::Evaluate(const arma::mat& coordinates) const
{
  double objective = 0.0;
  for (size_t i = 0; i < NumFunctions(); ++i)
  {
    objective += coordinates[i] * coordinates[i] + bi[i] * coordinates[i] +
      intercepts[i];
  }

  return objective;
}

//! Evaluate the gradient of a function.
inline void SparseTestFunction::Gradient(const arma::mat& coordinates,
                                         const size_t i,
                                         arma::sp_mat& gradient,
                                         const size_t batchSize) const
{
  gradient.zeros(arma::size(coordinates));
  for (size_t j = i; j < i + batchSize; ++j)
    gradient[j] = 2 * coordinates[j] + bi[j];
}

//! Evaluate the gradient of a feature function.
inline void SparseTestFunction::PartialGradient(const arma::mat& coordinates,
                                                const size_t j,
                                                arma::sp_mat& gradient) const
{
  gradient.zeros(arma::size(coordinates));
  gradient[j] = 2 * coordinates[j] + bi[j];
}

} // namespace test
} // namespace ens

#endif
