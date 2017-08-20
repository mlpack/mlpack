/**
 * @file matrix_completion_impl.hpp
 * @author Stephen Tu
 *
 * Implementation of MatrixCompletion class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "matrix_completion.hpp"

namespace mlpack {
namespace matrix_completion {

template<typename MCSolverType>
MatrixCompletion<MCSolverType>::
MatrixCompletion(const size_t m,
                 const size_t n,
                 const arma::umat& indices,
                 const arma::vec& values,
                 const size_t r) :
    m(m), n(n), indices(indices), values(values),
    mcSolver(m, n, indices, values, r)
{
  CheckValues();
}

template<typename MCSolverType>
MatrixCompletion<MCSolverType>::
MatrixCompletion(const size_t m,
                 const size_t n,
                 const arma::umat& indices,
                 const arma::vec& values,
                 const arma::mat& initialPoint) :
    m(m), n(n), indices(indices), values(values),
    mcSolver(m, n, indices, values, initialPoint)
{
  CheckValues();
}

template<typename MCSolverType>
MatrixCompletion<MCSolverType>::
MatrixCompletion(const size_t m,
                 const size_t n,
                 const arma::umat& indices,
                 const arma::vec& values) :
    m(m), n(n), indices(indices), values(values),
    mcSolver(m, n, indices, values)
{
  CheckValues();
}

template<typename MCSolverType>
void MatrixCompletion<MCSolverType>::CheckValues()
{
  if (indices.n_rows != 2)
  {
    Log::Fatal << "MatrixCompletion::CheckValues(): matrix of constraint "
        << "indices does not have 2 rows!" << std::endl;
  }

  if (indices.n_cols != values.n_elem)
  {
    Log::Fatal << "MatrixCompletion::CheckValues(): the number of constraint "
        << "indices (columns of constraint indices matrix) does not match the "
        << "number of constraint values (length of constraint value vector)!"
        << std::endl;
  }

  for (size_t i = 0; i < values.n_elem; i++)
  {
    if (indices(0, i) >= m || indices(1, i) >= n)
      Log::Fatal << "MatrixCompletion::CheckValues(): indices ("
          << indices(0, i) << ", " << indices(1, i)
          << ") are out of bounds for matrix of size " << m << " x n!"
          << std::endl;
  }
}

template<typename MCSolverType>
void MatrixCompletion<MCSolverType>::Recover(arma::mat& recovered)
{
    mcSolver.Recover(recovered, m, n);
}

} // namespace matrix_completion
} // namespace mlpack
