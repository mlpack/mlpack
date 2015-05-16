/**
 * @file laplacian_solver.hpp
 * @author Shangtong Zhang
 *
 * Calculate eigen vectors after transforming a matrix to laplacian matrix.
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EIGEN_SOLVER_LAPLACIAN_SOLVER
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EIGEN_SOLVER_LAPLACIAN_SOLVER

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for calculateing eigen vectors 
 * after transforming a matrix to laplacian matrix
 */
template<
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class LaplacianSolver
{
 public:
  /**
   * Calculate eigen vectors.
   * Equation (S - M) * eigenvector = eigenvalue * S * eigenvector will be solved.
   *
   * @param M desired matrix
   * @param eigvec store eigen vectors
   * @param eigval store eigen values
   */
  static void Solve(const MatType& M, MatType& eigvec, VecType& eigval)
  {
    arma::rowvec sumM = sum(M, 0);
    MatType S(M.n_rows, M.n_cols);
    S.zeros();
    S.diag() = sumM;
    MatType L = S - M;
    arma::cx_colvec cxEigval;
    arma::cx_mat cxEigvec;
    arma::eig_gen(cxEigval, cxEigvec, S.i() * L);
    eigval = arma::real(cxEigval);
    eigvec = arma::real(cxEigvec);
  }

};
    
}; // namespace manifold
}; // namespace mlpack

#endif
