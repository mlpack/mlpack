/**
 * @file le_transform.hpp
 * @author Shangtong Zhang
 *
 * Implementation for transforming similarity matrix for LE
 */
#ifndef __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_LE_TRANSFORM
#define __MLPACK_METHODS_MANIFOLD_LEARNING_TRANSFORM_RULE_LE_TRANSFORM

#include <mlpack/core.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>

namespace mlpack {
namespace manifold {
  
/**
 * This class is an implementation for transforming similarity matrix for LE
 * @tparam KernelType Kernel for applying transformation
 */
template<
    typename KernelType = kernel::GaussianKernel,
    typename SimilarityMatType = arma::sp_mat,
    typename MatType = arma::mat>
class LETransform
{
 public:
  
  /**
   * Create a LETransform object and set the parameters.
   * @param useNNKernel If true, nearest neighbor kernel will be used, thus
   *    @tparam KernelType and @param kernel will be ignored. It just sets
   *    M(i, j) to 1 if point i and point j are neighbors or 0 if they aren't 
   *    neighbors.
   * @param kernel Kernel to apply tranformation
   */
  LETransform(bool useNNKernel = true,
              const KernelType& kernel = KernelType()) :
      kernel(kernel), useNNKernel(useNNKernel)
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
    gramM.set_size(M.n_rows, M.n_cols);
    
    for (size_t i = 0; i < M.n_rows; ++i)
      for (size_t j = 0; j < M.n_cols; ++j)
      {
        if (M(i, j) == 0)
          gramM(i, j) = 0;
        else
          gramM(i, j) = useNNKernel ? 1 : kernel.Evaluate(M(i, j));
      }
    
    return gramM;
  }
  
  //! Get kernel
  KernelType& Kernel() { return kernel; }
  //! Modify kernel
  KernelType& Kernel() const { return kernel; }
  
  //! Get transformed similarity matrix
  MatType& GramM() const { return gramM; }
  //! Modify transformed similarity matrix
  MatType& GramM() { return gramM; }
  
  //! If use nearest neighbor kernel
  bool NNKernel() const { return useNNKernel; }
  
 private:
  //! Locally-stored kernel
  KernelType kernel;
  
  //! If use nearest neighbor kernel
  bool useNNKernel;
  
  //! Locally-stored transformed similarity matrix
  MatType gramM;
  
};
  
}; // namespace manifold
}; // namespace mlpack

#endif
