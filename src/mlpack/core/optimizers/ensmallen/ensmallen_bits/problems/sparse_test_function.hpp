/**
 * @file sparse_test_function.hpp
 * @author Shikhar Bhardwaj
 *
 * Sparse test function for Parallel SGD.
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_HPP
#define ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_HPP

namespace ens {
namespace test {

// A simple test function. Each dimension has a parabola with a
// distinct minimum. Each update is guaranteed to be sparse(only a single
// dimension is updated in the decision variable by each thread). At the end of
// a reasonable number of iterations, each value in the decision variable should
// be at the vertex of the parabola in that dimension.
class SparseTestFunction
{
 public:
  //! Set members in the default constructor.
  SparseTestFunction();

  //! Return 4 (the number of functions).
  size_t NumFunctions() const { return 4; }

  //! Return 4 (the number of features).
  size_t NumFeatures() const { return 4; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("0 0 0 0;"); }

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates,
                  const size_t i,
                  const size_t batchSize = 1) const;

  //! Evaluate all the functions.
  double Evaluate(const arma::mat& coordinates) const;

  //! Evaluate the gradient of a function.
  void Gradient(const arma::mat& coordinates,
                const size_t i,
                arma::sp_mat& gradient,
                const size_t batchSize = 1) const;

  //! Evaluate the gradient of a feature function.
  void PartialGradient(const arma::mat& coordinates,
                       const size_t j,
                       arma::sp_mat& gradient) const;

 private:
  // Each quadratic polynomial is monic. The intercept and coefficient of the
  // first order term is stored.

  //! The vector storing the intercepts
  arma::vec intercepts;

  //! The vector having coefficients of the first order term
  arma::vec bi;
};

} // namespace test
} // namespace ens

// Include implementation.
#include "sparse_test_function_impl.hpp"

#endif // ENSMALLEN_PROBLEMS_SPARSE_TEST_FUNCTION_HPP
