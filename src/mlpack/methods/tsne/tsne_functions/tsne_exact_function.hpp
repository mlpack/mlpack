/**
 * @file methods/tsne/tsne_functions/tsne_exact_function.hpp
 * @author Ranjodh Singh
 *
 * Definition of the exact objective function for t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>

#include "../tsne_utils.hpp"

namespace mlpack {

/**
 * Exact objective function for t-SNE.
 *
 * @note The specified distance metric is used only for computing input
 * space similarities. Similarities in the embedding space are computed
 * using the Euclidean distance via the Student's t-distribution kernel.
 *
 * @tparam MatType The type of Matrix.
 * @tparam DistanceType The distance metric for computing input space
 *     similarities.
 */
template <typename MatType, typename DistanceType>
class TSNEExactFunction
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;

  /**
   * Constructs the TSNEExactFunction object.
   *
   * @param X The input data.
   * @param perplexity Perplexity of the input space similarity distribution.
   * @param dof Degrees of freedom.
   * @param theta Unused parameter, present for compatibility.
   */
  TSNEExactFunction(const MatType& X,
                    const double perplexity,
                    const size_t dof,
                    const double /* theta */);

  /**
   * EvaluateWithGradient for differentiable function optimizers.
   *
   * Returns the Kullback-Leibler (KL) divergence between the input and
   * the embedding, and stores its gradient w.r.t. the embedding in `g`.
   *
   * @param y The embedding matrix.
   * @param g The variable to store the gradient.
   *
   * @return The KL divergence value.
   */
  template <typename GradType>
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& y, GradType& g);

  //! Get the input joint probabilities.
  const MatType& InputJointProbabilities() const { return P; }
  //! Modify the input joint probabilities.
  MatType& InputJointProbabilities() { return P; }

 private:
  //! Input space similarity distribution (normalized).
  MatType P;

  //! Embedding space similarity distribution. (unnormalized)
  MatType q;

  //! Embedding space similarity distribution. (normalized)
  MatType Q;

  //! Matrix holding an intermediate term for gradient computation.
  MatType deltaPQ;

  //! Perplexity of the input space similarity distribution.
  double perplexity;

  //! Degrees of freedom.
  size_t dof;
};

} // namespace mlpack

// Include implementation.
#include "./tsne_exact_function_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP
