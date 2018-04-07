/**
 * @file sparse_test_function.hpp
 * @author Shikhar Bhardwaj
 *
 * Sparse test function for Parallel SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_TEST_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_PARALLEL_SGD_SPARSE_TEST_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {
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
  SparseTestFunction()
  {
    intercepts = arma::vec("20 12 15 100");
    bi = arma::vec("-4 -2 -3 -8");
  }

  //! Return 4 (the number of functions).
  size_t NumFunctions() const { return 4; }

  //! Return 4 (the number of features).
  size_t NumFeatures() const { return 4; }

  //! Get the starting point.
  arma::mat GetInitialPoint() const { return arma::mat("0 0 0 0;"); }

  //! Evaluate a function.
  double Evaluate(const arma::mat& coordinates,
                  const size_t i,
                  const size_t batchSize = 1) const
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
  double Evaluate(const arma::mat& coordinates) const
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
  void Gradient(const arma::mat& coordinates,
                const size_t i,
                arma::sp_mat& gradient,
                const size_t batchSize = 1) const
  {
    gradient.zeros(arma::size(coordinates));
    for (size_t j = i; j < i + batchSize; ++j)
      gradient[j] = 2 * coordinates[j] + bi[j];
  }

  //! Evaluate the gradient of a feature function.
  void PartialGradient(const arma::mat& coordinates,
                       const size_t j,
                       arma::sp_mat& gradient) const
  {
    gradient.zeros(arma::size(coordinates));
    gradient[j] = 2 * coordinates[j] + bi[j];
  }

 private:
  // Each quadratic polynomial is monic. The intercept and coefficient of the
  // first order term is stored.

  //! The vector storing the intercepts
  arma::vec intercepts;

  //! The vector having coefficients of the first order term
  arma::vec bi;
};

} // namespace test
} // namespace optimization
} // namespace mlpack

#endif
