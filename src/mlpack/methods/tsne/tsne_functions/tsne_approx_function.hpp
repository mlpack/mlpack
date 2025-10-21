/**
 * @file methods/tsne/tsne_function/tsne_approx_function.hpp
 * @author Ranjodh Singh
 *
 * t-SNE Approx Function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP

#include <mlpack/core/tree/octree/octree.hpp>
#include <mlpack/core/util/arma_traits.hpp>
#include <omp.h>
#include <armadillo>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/methods/neighbor_search.hpp>
#include <type_traits>

#include "../tsne_utils.hpp"
#include "../tsne_rules/tsne_rules.hpp"

namespace mlpack {

/**
 * Approximate gradient of the KL-divergence objective, designed for
 * optimization with ensmallen.
 *
 * This class implements an tree-based approximation of the t-SNE objective
 * that decomposes forces into positive (attractive) and negative (repulsive)
 * components and computes the gradient efficiently using an octree. It is
 * templated on whether to use a dual-tree or the barnes-hut method.
 *
 * @tparam UseDualTree Indicates whether the traversal is dual (true) or
           single (false). Allows both barnes-hut and dual-tree approximations
           to be handled in one class.
 * @tparam DistanceType The distance metric to use for computation.
 * @tparam MatType The type of Matrix.
 */
template <bool UseDualTree,
          typename DistanceType = SquaredEuclideanDistance,
          typename MatType = arma::mat>
class TSNEApproxFunction
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;
  using SpMatType = typename GetSparseMatType<MatType>::type;
  using RuleType = TSNERules<MatType>;
  using TreeType = Octree<SquaredEuclideanDistance, CentroidStatistic>;

  /**
   * Constructs the TSNEApproxFunction object.
   *
   * @param X The input data. (Din X N)
   * @param perplexity The perplexity of the Gaussian distribution.
   * @param dof The degrees of freedom.
   * @param theta The coarseness of the approximation.
   */
  TSNEApproxFunction(const MatType& X,
                     const double perplexity,
                     const size_t dof,
                     const double theta = 0.5);

  /**
   * EvaluateWithGradient for differentiable function optimizers
   * Evaluates the Kullback–Leibler (KL) divergence between input
   * and the embedding and updates gradients.
   *
   * @param y Current embedding.
   * @param g Variable to store the new gradient.
   */
  template <typename GradType>
  double EvaluateWithGradient(const MatType& y, GradType& g);

  /**
   * Calculates the negative (repulsive) component of the gradient
   * using dual tree approximation (enabled via tag dispatch).
   * I will calculate the normalized negative gradient and subtract
   * it from g and also return the normalization value.
   *
   * @param g Gradient matrix.
   * @param y Embedding matrix
   *
   * @return The normalization value for the negative gradient.
   */
  double CalculateNegativeGradient(
      MatType &g, const MatType& y, std::true_type /* tag */);

  /**
   * Calculates the negative (repulsive) component of the gradient
   * using barnes hut approximation (enabled via tag dispatch).
   * I will calculate the normalized negative gradient and subtract
   * it from g and also return the normalization value.

   * @param g Gradient matrix.
   * @param y Embedding matrix.
   *
   * @return The normalization value for the negative gradient.
   */
  double CalculateNegativeGradient(
      MatType &g, const MatType& y, std::false_type /* tag */);

  /**
   * Calculates the positive (attractive) component of the gradient
   * and adds it to the total gradient.
   * It will calculate the positive gradient term and add it to g.
   * and also return the kl divergence value.
   *
   * @param g Matrix to store the positive gradient.
   * @param y Current embedding.
   * @param sumQ The normalization value for the negative gradient.
   *
   * @return The KL Divergence Value.
   */
  double CalculatePositiveGradient(
    MatType &g, const MatType& y, const double sumQ);

  //! Get the Input Joint Probabilities.
  const SpMatType& InputJointProbabilities() const { return P; }
  //! Modify the Input Joint Probabilities.
  SpMatType& InputJointProbabilities() { return P; }

 private:
  //! Input joint probabilities.
  SpMatType P;

  //! Nearest neibhbor distances.
  MatType D;

  //! Nearest neighbor indexes.
  arma::Mat<size_t> N;

  //! The perplexity of the Gaussian distribution.
  double perplexity;

  //! Degrees of freedom.
  size_t dof;

  //! The coarseness of the approximation.
  double theta;
};

} // namespace mlpack

// Include implementation.
#include "./tsne_approx_function_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
