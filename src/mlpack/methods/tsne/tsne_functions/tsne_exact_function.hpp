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
#include <mlpack/core/cv/metrics/facilities.hpp>

#include "../tsne_utils.hpp"

namespace mlpack {

/**
 * Exact objective function for t-SNE.
 *
 * @tparam MatType The type of Matrix.
 * @tparam DistanceType The distance metric.
 */
template <typename MatType,
          typename DistanceType>
class TSNEExactFunction
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;

  /**
   * Constructs the TSNEExactFunction object.
   *
   * @param X The input data.
   * @param perplexity Perplexity of the Gaussian distribution.
   * @param dof Degrees of freedom.
   */
  TSNEExactFunction(const MatType& X,
                    const double perplexity,
                    const size_t dof,
                    const double /* theta */);

  /**
   * EvaluateWithGradient for differentiable function optimizers.
   *
   * Returns the Kullback-Leibler (KL) divergence between the input and
   * the embedding, and stores its gradient w.r.t to the embedding in `g`.
   *
   * @param y The embedding matrix.
   * @param g The variable used to store the new gradient.
   */
  template <typename GradType>
  typename MatType::elem_type EvaluateWithGradient(
      const MatType& y, GradType& g);

  //! Get the input joint probabilities.
  const MatType& InputJointProbabilities() const { return P; }
  //! Modify the input joint probabilities.
  MatType& InputJointProbabilities() { return P; }

 private:
  //! Input joint probabilities.
  MatType P;

  //! Output joint probabilities. (Unnormalized)
  MatType q;

  //! Output joint probabilities. (Normalized)
  MatType Q;

  //! Intermediate matrix used in gradient computation.
  MatType M;

  //! Intermediate vector used in gradient computation.
  VecType S;

  //! Perplexity of the Gaussian distribution.
  double perplexity;

  //! Degrees of freedom.
  size_t dof;
};

} // namespace mlpack

// Include implementation.
#include "./tsne_exact_function_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP
