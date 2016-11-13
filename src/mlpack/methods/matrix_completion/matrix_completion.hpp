/**
 * @file matrix_completion.hpp
 * @author Stephen Tu
 *
 * A thin wrapper around nuclear norm minimization to solve
 * low rank matrix completion problems.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP

#include <mlpack/core/optimizers/sdp/sdp.hpp>
#include <mlpack/core/optimizers/sdp/lrsdp.hpp>

namespace mlpack {
namespace matrix_completion {

/**
 * This class implements the popular nuclear norm minimization heuristic for
 * matrix completion problems. That is, given known values M_ij's, the
 * following optimization problem (semi-definite program) is solved to fill in
 * the remaining unknown values of X
 *
 *   min ||X||_* subj to X_ij = M_ij
 *
 * where ||X||_* denotes the nuclear norm (sum of singular values of X).
 *
 * For a theoretical treatment of the conditions necessary for exact recovery,
 * see the following paper:
 *
 *   A Simpler Appoarch to Matrix Completion.
 *   Benjamin Recht. JMLR 11.
 *   http://arxiv.org/pdf/0910.0651v2.pdf
 *
 * An example of how to use this class is shown below:
 *
 * @code
 * size_t m, n;         // size of unknown matrix
 * arma::umat indices;  // contains the known indices [2 x n_entries]
 * arma::vec values;    // contains the known values [n_entries]
 * arma::mat recovered; // will contain the completed matrix
 *
 * MatrixCompletion mc(m, n, indices, values);
 * mc.Recover(recovered);
 * @endcode
 *
 * @see LRSDP
 */
class MatrixCompletion
{
 public:
  /**
   * Construct a matrix completion problem, specifying the maximum rank of the
   * solution.
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
   * @param m Number of rows of original matrix.
   * @param n Number of columns of original matrix.
   * @param indices Matrix containing the indices of the known entries (must be
   *    [2 x p]).
   * @param values Vector containing the values of the known entries (must be
   *    length p).
   * @param initialPoint Starting point for the SDP optimization.
   */
  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values,
                   const arma::mat& initialPoint);

  /**
   * Construct a matrix completion problem.
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
   * Solve the underlying SDP to fill in the remaining values.
   *
   * @param recovered Will contain the completed matrix.
   */
  void Recover(arma::mat& recovered);

  //! Return the underlying SDP.
  const optimization::LRSDP<optimization::SDP<arma::sp_mat>>& Sdp() const { return sdp; }
  //! Modify the underlying SDP.
  optimization::LRSDP<optimization::SDP<arma::sp_mat>>& Sdp() { return sdp; }

 private:
  //! Number of rows in original matrix.
  size_t m;
  //! Number of columns in original matrix.
  size_t n;
  //! Matrix containing the indices of the known entries (has two rows).
  arma::umat indices;
  //! Vector containing the values of the known entries.
  arma::mat values;

  //! The underlying SDP to be solved.
  optimization::LRSDP<optimization::SDP<arma::sp_mat>> sdp;

  //! Validate the input matrices.
  void CheckValues();
  //! Initialize the SDP.
  void InitSDP();

  //! Select a rank of the matrix given that is of size m x n and has p known
  //! elements.
  static size_t DefaultRank(const size_t m, const size_t n, const size_t p);
};

} // namespace matrix_completion
} // namespace mlpack

#endif
