/**
 * @file mc_sdp_solver.hpp
 * @author Stephen Tu
 * @author Chenzhe Diao
 *
 * Matrix Completion using SDP solver.
 * Modified from the original code by Stephen Tu.
 *
 * A thin wrapper around nuclear norm minimization to solve
 * low rank matrix completion problems using SPD solver.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_SDP_SOLVER_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_SDP_SOLVER_HPP

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
 * @see LRSDP
 */
class MCSDPSolver
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
  MCSDPSolver(const size_t m,
              const size_t n,
              const arma::umat& indices,
              const arma::vec& values,
              const size_t r) :
      sdp(indices.n_cols, 0, arma::randu<arma::mat>(m + n, r))
  { InitSDP(m, n, indices, values); }

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
   * @param initialPoint Starting point of the sdp optimization.
   */
  MCSDPSolver(const size_t m,
              const size_t n,
              const arma::umat& indices,
              const arma::vec& values,
              const arma::mat& initialPoint) :
      sdp(indices.n_cols, 0, initialPoint)
  { InitSDP(m, n, indices, values); }

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
  MCSDPSolver(const size_t m,
              const size_t n,
              const arma::umat& indices,
              const arma::vec& values) :
      sdp(indices.n_cols, 0,
          arma::randu<arma::mat>(m + n, DefaultRank(m, n, indices.n_cols)))
  { InitSDP(m, n, indices, values); }

  //! This type of constructor is not feasible for SDP solver.
  MCSDPSolver(const size_t m,
              const size_t n,
              const arma::umat& indices,
              const arma::vec& values,
              const double tau) :
      sdp(indices.n_cols, 0,
          arma::randu<arma::mat>(m + n, DefaultRank(m, n, indices.n_cols)))

    {
      Log::Fatal << "No such constructor for SDP solver!" << std::endl;
    }

  /**
   * Solve the underlying sdp to fill in the remaining values.
   *
   * @param recovered Will contain the completed matrix.
   */
  void Recover(arma::mat& recovered, const size_t m, const size_t n)
  {
    recovered = sdp.Function().GetInitialPoint();
    sdp.Optimize(recovered);
    recovered = recovered * trans(recovered);
    recovered = recovered(arma::span(0, m - 1), arma::span(m, m + n - 1));
  }

private:
    //! The underlying SDP to be solved.
    optimization::LRSDP<optimization::SDP<arma::sp_mat>> sdp;
    
    size_t DefaultRank(const size_t m, const size_t n, const size_t p)
    {
        // If r = O(sqrt(p)), then we are guaranteed an exact solution.
        // For more details, see
        //
        //   On the rank of extreme matrices in semidefinite programs and the
        //   multiplicity of optimal eigenvalues.
        //   Pablo Moscato, Michael Norman, and Gabor Pataki.
        //   Math Oper. Res., 23(2). 1998.
        const size_t mpn = m + n;
        float r = 0.5 + sqrt(0.25 + 2 * p);
        if (ceil(r) > mpn)
            r = mpn; // An upper bound on the dimension.
        return ceil(r);
    }
    
    void InitSDP(const size_t m, const size_t n,
                 const arma::umat& indices,
                 const arma::vec& values)
    {
        sdp.SDP().C().eye(m + n, m + n);
        sdp.SDP().SparseB() = 2. * values;
        const size_t p = indices.n_cols;
        for (size_t i = 0; i < p; i++)
        {
            sdp.SDP().SparseA()[i].zeros(m + n, m + n);
            sdp.SDP().SparseA()[i](indices(0, i), m + indices(1, i)) = 1.;
            sdp.SDP().SparseA()[i](m + indices(1, i), indices(0, i)) = 1.;
        }
    }

};
}
}

#endif

