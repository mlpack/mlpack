/**
 * @file mds_transform.hpp
 * @author Shangtong Zhang
 *
 * Implementation for transforming similarity matrix for MDS
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_MDS_TRANSFORM
#define __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_MDS_TRANSFORM

#include <mlpack/core.hpp>

namespace mlpack {
namespace manifold {

/**
 * This class is an implementation for transforming similarity matrix for MDS
 */
template<
    typename MatType = arma::mat>
class MDSTransform
{
 public:
  
  /**
   * Transform similarity matrix
   * @param M Similarity matrix to be transformed
   * @return reference for transformed similarity matrix
   */
  MatType& Transform(const MatType& M)
  {
    size_t nData = M.n_rows;
    arma::rowvec S = arma::sum(M);
    gramM.zeros(nData, nData);
    double sumS = sum(S) / nData / nData;
    
    for (size_t i = 0; i < nData; ++i)
      for (size_t j = 0; j < nData; ++j)
        gramM(i, j) = -0.5 * (M(i, j) - S(i) / nData - S(j) / nData + sumS);
    return gramM;
  }
  
  //! Get transformed similarity matrix
  MatType& GramM() const { return gramM; }
  //! Modify transformed similarity matrix
  MatType& GramM() { return gramM; }
  
 private:
  //! Locally-stored transformed similarity matrix
  MatType gramM;
  
};

}; // namespace manifold
}; // namespace mlpack

#endif
