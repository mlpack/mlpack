/**
 * @file matrix_completion.hpp
 * @author Stephen Tu
 *
 * A thin wrapper around nuclear norm minimization to solve
 * low rank matrix completion problems.
 */
#ifndef __MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP
#define __MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP

#include <mlpack/core/optimizers/lrsdp/lrsdp.hpp>

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
 * size_t m, n;        // size of unknown matrix
 * arma::umat indices; // contains the known indices [2 x n_entries]
 * arma::vec values;   // contains the known values [n_entries]
 *
 * MatrixCompletion mc(m, n, indices, values);
 * mc.Recover();
 * mc.Recovered();     // access completed matrix
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
   */
  void Recover();

  //! Return the underlying SDP.
  const optimization::LRSDP& Sdp() const { return sdp; }
  //! Modify the underlying SDP.
  optimization::LRSDP& Sdp() { return sdp; }

  //! Return the recovered matrix.
  const arma::mat& Recovered() const { return recovered; }

 private:
  size_t m;

  size_t n;

  arma::umat indices;

  arma::mat values;

  optimization::LRSDP sdp;

  arma::mat recovered;

  void checkValues();

  void initSdp();

  static size_t
  DefaultRank(const size_t m,
              const size_t n,
              const size_t p);

  static arma::mat
  CreateInitialPoint(const size_t m,
                     const size_t n,
                     const size_t r);

};

} // namespace matrix_completion
} // namespace mlpack

// Include implementation.
#include "matrix_completion_impl.hpp"

#endif
