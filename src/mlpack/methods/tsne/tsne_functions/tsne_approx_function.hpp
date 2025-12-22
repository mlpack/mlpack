/**
 * @file methods/tsne/tsne_function/tsne_approx_function.hpp
 * @author Ranjodh Singh
 *
 * Definition of the approximate objective function for t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP

#include <omp.h>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/methods/neighbor_search.hpp>

#include "../tsne_utils.hpp"
#include "../tsne_rules/tsne_rules.hpp"

namespace mlpack {

/**
 * Approximate objective function for t-SNE.
 *
 * This class implements an tree-based approximation of the t-SNE objective
 * that decomposes forces into attractive and repulsive components and computes
 * the gradient efficiently using an octree. The implementation is templated to
 * support both the Dual-Tree and the Barnes-Hut method.
 *
 * @tparam UseDualTree Flag indicating whether to use the Dual-Tree method
 *         instead of the Barnes-Hut method.
 * @tparam MatType The type of Matrix.
 * @tparam DistanceType The distance metric.
 */
template <bool UseDualTree,
          typename MatType,
          typename DistanceType>
class TSNEApproxFunction
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using VecType = typename GetColType<MatType>::type;
  using SpMatType = typename GetSparseMatType<MatType>::type;
  using RuleType = TSNERules<MatType>;
  using TreeType = Octree<SquaredEuclideanDistance,
                          CentroidStatisticType<VecType>,
                          MatType>;

  /**
   * Constructs the TSNEApproxFunction object.
   *
   * @param X The input data.
   * @param perplexity Perplexity of the Gaussian distribution.
   * @param dof Degrees of freedom.
   * @param theta Coarseness of the approximation.
   */
  TSNEApproxFunction(const MatType& X,
                     const double perplexity,
                     const size_t dof,
                     const double theta);

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
  double EvaluateWithGradient(const MatType& y, GradType& g);

  /**
   * Calculates the repulsive part of the gradient using Dual-Tree
   * approximation.
   *
   * @param g The gradient matrix.
   * @param y The embedding matrix
   *
   * @return The normalization value for the repulsive forces.
   */
  double CalculateRepuliveForces(
      MatType &g, const MatType& y, std::true_type /* tag */);

  /**
   * Calculates the repulsive part of the gradient using Barnes-Hut
   * approximation.
   *
   * @param g The gradient matrix.
   * @param y The embedding matrix.
   *
   * @return The normalization value for the repulsive forces.
   */
  double CalculateRepuliveForces(
      MatType &g, const MatType& y, std::false_type /* tag */);

  /**
   * Calculates the attractive part of the gradient.
   *
   * @param g The gradient matrix.
   * @param y The embedding matrix.
   * @param sumQ The normalization value for the repulsive forces.
   *
   * @return The KL Divergence Value.
   */
  double CalculateAttractiveForces(
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

  //! Nearest neighbor indices.
  arma::Mat<size_t> N;

  //! Perplexity of the Gaussian distribution.
  double perplexity;

  //! Degrees of freedom.
  size_t dof;

  //! Coarseness of the approximation.
  double theta;
};

// Convenience typedef for TSNEDualTreeFunction
template<typename MatType, typename DistanceType>
using TSNEDualTreeFunction = TSNEApproxFunction<true, MatType, DistanceType>;

// Convenience typedef for TSNEBarnesHutFunction
template<typename MatType, typename DistanceType>
using TSNEBarnesHutFunction = TSNEApproxFunction<false, MatType, DistanceType>;

} // namespace mlpack

// Include implementation.
#include "./tsne_approx_function_impl.hpp"

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
