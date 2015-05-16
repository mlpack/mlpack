/**
 * @file standard_solver.hpp
 * @author Shangtong Zhang
 *
 * Calculate eigen vectors directly.
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EIGEN_SOLVER_STANDARD_SOLVER
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EIGEN_SOLVER_STANDARD_SOLVER

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for calculateing eigen vectors directly.
 * @tparam Symmetric If the matrix is symmetric
 */
template<
    bool Symmetric = false,
    typename MatType = arma::mat,
    typename VecType = arma::colvec>
class StandardSolver
{
 public:
  
  /**
   * Calculate eigen vectors for symmetric matrix
   * @param M desired matrix
   * @param eigvec store eigen vectors
   * @param eigval store eigen values
   */
  template<bool Sym = Symmetric>
  static typename std::enable_if<Sym, void>::type
  Solve(const MatType& M, MatType& eigvec, VecType& eigval)
  {
    arma::eig_sym(eigval, eigvec, M);
  }
  
  /**
   * Calculate eigen vectors for general matrix
   * @param M desired matrix
   * @param eigvec store eigen vectors
   * @param eigval store eigen values
   */
  template<bool Sym = Symmetric>
  static typename std::enable_if<!Sym, void>::type
  Solve(const MatType& M, MatType& eigvec, VecType& eigval)
  {
    arma::cx_vec cxEigval;
    arma::cx_mat cxEigvec;
    arma::eig_gen(cxEigval, cxEigvec, M);
    eigval = arma::real(cxEigval);
    eigvec = arma::real(cxEigvec);
  }
};
    
}; // namespace manifold
}; // namespace mlpack

#endif
