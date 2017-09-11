/**
 * @file matrix_completion.hpp
 * @author Stephen Tu
 * @author Chenzhe Diao
 *
 * A thin wrapper around different types of solvers for
 * low rank matrix completion problems.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP

#include "mc_sdp_solver.hpp"
#include "mc_fw_solver.hpp"

namespace mlpack {
namespace matrix_completion {

/**
 * This class implements a wrapper for different types of solvers of matrix
 * completion problems. That is, we want to recover a large low rank matrix,
 * with only a few elements in the matrix known. There are quite many
 * mathematical models and numerical algorithms to solve this kind of problems.
 * This class only provide a wrapper using class templates.
 *
 * The model solved using SDP solver is:
 *
 * \f[
 *   min ||X||_* subj to X_ij = M_ij, \qquad (i,j) \in \Omega
 * \f]
 *
 * where ||X||_* denotes the nuclear norm (sum of singular values of X).
 *
 * For a theoretical treatment of the conditions necessary for exact recovery
 * using the above model, see the following paper:
 *
 *   A Simpler Appoarch to Matrix Completion.
 *   Benjamin Recht. JMLR 11.
 *   http://arxiv.org/pdf/0910.0651v2.pdf
 *
 * The model solved using FrankWolfe solver is:
 *
 * \f[
 * min \sum_{(i,j) \in \Omega} 0.5 * (X_ij - M_ij)^2, \qquad
 * s.t. ||X||_* <= tau
 * \f]
 *
 *
 * Example of the usage of the code is:
 *
 * @code
 * size_t m, n;         // size of unknown matrix
 * arma::umat indices;  // contains the known indices [2 x n_entries]
 * arma::vec values;    // contains the known values [n_entries]
 * arma::mat recovered; // will contain the completed matrix
 *
 * MatrixCompletion<MCSolverType> mc(m, n, indices, values);
 * mc.Recover(recovered);
 * @endcode
 *
 */
template<typename MCSolverType>
class MatrixCompletion
{
 public:
  /**
   * Construct a matrix completion problem, specifying the maximum rank of the
   * solution.
   *
   * This constructor could be used in MatrixCompletionSDP class.
   *
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   * @param r Maximum rank of solution.
   */
  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values,
                   const size_t r);

  /**
   * Construct a matrix completion problem, specifying the initial point of the
   * optimization.
   *
   * This constructor could be used in MatrixCompletionSDP class.
   *
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   * @param initialPoint Starting point optimization.
   */
  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values,
                   const arma::mat& initialPoint);


  /**
   * Construct a matrix completion problem.
   *
   * This constructor could be used in MatrixCompletionSDP class.
   *
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   */
  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values);

  /**
   * Construct a matrix completion problem.
   *
   * This constructor could be used in MatrixCompletionFW class.
   *
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   * @param tau Maximum nuclear norm constraint of the solution.
   */
  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values,
                   const double tau);

  /**
   * Solve the underlying optimization problem to fill in the remaining values.
   *
   * @param recovered Will contain the completed matrix.
   */
  void Recover(arma::mat& recovered);

  //! Return the underlying matrix completion solver.
  const MCSolverType& MCSolver() const { return mcSolver; }
  //! Modify the underlying matrix completion solver.
  MCSolverType& MCSolver() { return mcSolver; }

 private:
  //! Number of rows in original matrix.
  size_t m;
  //! Number of columns in original matrix.
  size_t n;

  //! The underlying matrix completion optimization solver.
  MCSolverType mcSolver;

  //! Validate the input matrices.
  void CheckValues(const arma::umat& indices, const arma::vec& values);
};

//! Matrix Completion using SDP Solver.
using MatrixCompletionSDP = MatrixCompletion<MCSDPSolver>;

//! Matrix Completion using Frank-Wolfe Solver.
using MatrixCompletionFW  = MatrixCompletion<MCFWSolver>;

} // namespace matrix_completion
} // namespace mlpack

#include "matrix_completion_impl.hpp"

#endif
