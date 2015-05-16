/**
 * @file lle_transform.hpp
 * @author Shangtong Zhang
 *
 * Implementation for transforming similarity matrix for LLE
 */

#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_LLE_TRANSFORM
#define __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_LLE_TRANSFORM

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for transforming similarity matrix for LLE
 */
template<
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class LLETransform
{
 public:
  
  /**
   * Create a LLETransform object and set the parameters.
   * @param mu Parameter for transformation.
   */
  LLETransform(double mu = 1e4):
      mu(mu)
  {
    /* nothing to do */
  }
  
  /**
   * Transform similarity matrix
   * @param M Similarity matrix to be transformed
   * @return reference for transformed similarity matrix
   */
  MatType& Transform(const SimilarityMatType& M)
  {
    MatType identity = arma::eye<MatType>(M.n_rows, M.n_cols);
    MatType tmp = identity - M;
    gramM = tmp.t() * tmp;
    gramM = mu * identity - gramM;
    return gramM;
  }
  
  //! Get transformed similarity matrix
  MatType& GramM() const { return gramM; }
  //! Modify transformed similarity matrix
  MatType& GramM() { return gramM; }
  
  //! Get mu
  double Mu() const { return mu; }
  //! Modify mu
  double& Mu() { return mu; }
  
 private:
  //! Locally-stored transformed similarity matrix
  MatType gramM;
  
  //! Locally-stored mu
  double mu;
  
};
    
}; // namespace manifold
}; // namespace mlpack

#endif
