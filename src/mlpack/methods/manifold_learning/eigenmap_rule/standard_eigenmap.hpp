/**
 * @file standard_eigenmap.hpp
 * @author Shangtong Zhang
 *
 * Implementation for calculating eigen vectors and map them to 
 * embedding vectors.
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_EIGENMAP_RULE_STANDARD_EIGENMAP
#define __MLPACK_METHODS_MANIFOLD_LEARNING_EIGENMAP_RULE_STANDARD_EIGENMAP

#include <mlpack/core.hpp>
#include <mlpack/methods/manifold_learning/eigen_solver/standard_solver.hpp>
#include <mlpack/methods/manifold_learning/eigen_solver/laplacian_solver.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for calculating eigen vectors from transformed
 * similarity matrix and map them to embedding vectors.
 *
 * @tparam SolverType How to get eigen vectors from transformed similarity matrix.
 * @tparam SkippedEigvec # of eigen vectors to be skipped
 * @tparam UseSqrtEigval If true, eigen vector will be multiplied by square root of
 *    of corresponding eigen value when mapping.
 */
  
template<
    typename SolverType,
    size_t SkippedEigvec = 0,
    bool UseSqrtEigval = true,
    typename MatType = arma::mat>
class StandardEigenmap
{
 public:
  
  /**
   * Calculating embedding vectors.
   * @param M Tranformed similarity matrix
   * @param embeddingMat Store embedding vectors, each column represents a point
   * @dim dimension of embedding vectors
   */
  void Eigenmap(const MatType& M,
                MatType& embeddingMat,
                size_t dim)
  {
    // Get eigen vectors and eigen values
    arma::colvec eigval;
    arma::mat eigvec;
    SolverType::Solve(M, eigvec, eigval);
    
    // Sort eigen values in descending order
    arma::uvec ind = arma::sort_index(eigval, 1);
    
    orderedEigvec.set_size(M.n_cols, dim);
    orderedEigval.set_size(dim, 1);
    size_t dimCount = 0;
    
    // Map eigen vectors to embedding vectors
    for (size_t i = SkippedEigvec; i < ind.n_elem; ++i)
    {
      if (dimCount >= dim || eigval(ind(i)) < 0)
        break;
      orderedEigvec.col(dimCount) = eigvec.unsafe_col(ind(i));
      if (UseSqrtEigval)
        orderedEigvec.col(dimCount) *= std::sqrt(eigval(ind(i)));
      orderedEigval(dimCount) = eigval(ind(i));
      dimCount++;
    }
    embeddingMat = orderedEigvec.t();
  }
  
  //! Get eigen vectors
  MatType& Eigvec() const { return orderedEigvec; }
  //! Modify eigen vectors
  MatType& Eigvec() { return orderedEigvec; }
  
  //! Get eigen values
  arma::colvec& Eigval() const { return orderedEigval; }
  //! Modify eigen values
  arma::colvec& Eigval() { return orderedEigval; }
  
 private:
  //! Locally-stored eigen vectors
  MatType orderedEigvec;
  
  //! Locally-stored eigen values
  arma::colvec orderedEigval;
};
  
// typedef for convenience
  
// define eigenmap rule for MDS
template<typename MatType = arma::mat>
using MDSEigenmap = StandardEigenmap<
    StandardSolver<true, MatType>, 0, true, MatType>;

// define eigenmap rule for Isomap
template<typename MatType = arma::mat>
using IsomapEigenmap = StandardEigenmap<
    StandardSolver<false, MatType>, 0, true, MatType>;
  
// define eigenmap rule for LLE
template<typename MatType = arma::mat>
using LLEEigenmap = StandardEigenmap<
    StandardSolver<false, MatType>, 1, false, MatType>;

// define eigenmap rule for LE
template<typename MatType = arma::mat>
using LEEigenmap = StandardEigenmap<
    LaplacianSolver<MatType>, 1, false, MatType>;
  
}; // namespace manifold
}; // namespace mlpack

#endif
