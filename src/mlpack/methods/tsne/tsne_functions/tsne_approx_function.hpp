/**
 * @file methods/tsne/tsne_function/tsne_approx_function.hpp
 * @author Ranjodh Singh
 *
 * t-SNE Approx Function
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP

#include <armadillo>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/methods/neighbor_search.hpp>

#include "../tsne_utils.hpp"
#include "../tsne_approx_rules.hpp"
#include "../centroid_statistic.hpp"

namespace mlpack
{

template <bool UseDualTree, typename MatType = arma::mat>
class TSNEApproxFunction
{
 public:
  // Convenience typedefs.
  using DistanceType = SquaredEuclideanDistance;
  using TreeType = Octree<DistanceType, CentroidStatistic>;

  TSNEApproxFunction(const MatType& X,
                     const double perplexity,
                     const double theta = 0.5)
      : perplexity(perplexity), theta(theta)
  {
    degrees_of_freedom = std::max<size_t>(X.n_rows - 1, 1);

    // Run KNN
    NeighborSearch<NearestNeighborSort, DistanceType> knn(X);
    const size_t neighbors = static_cast<size_t>(3 * perplexity);
    knn.Search(neighbors, N, D);

    // Pre Compute P
    P = binarySearchPerplexity(perplexity, N, D);
    P = P + P.t();
    P /= std::max(arma::datum::eps, arma::accu(P));
  }

  /**
   * EvaluateWithGradient for differentiable function optimizers
   * Evaluates the Kullback–Leibler (KL) divergence between input
   * and the embedding and updates gradients.
   *
   * @param y Current embedding
   * @param g Variable to store the new gradient
   */
  template <typename GradType>
  double EvaluateWithGradient(const MatType& y, GradType& g)
  {
    // Init
    double sumQ = 0.0, error = 0.0;
    std::vector<size_t> oldFromNew;
    TreeType tree(y, oldFromNew, 1);
    TSNEApproxRules<UseDualTree> rule(sumQ, g, y, oldFromNew, theta);

    // Negative Force Calculation
    if constexpr (UseDualTree)
    {
      TreeType::DualTreeTraverser traverser(rule);
      traverser.Traverse(tree, tree);
    }
    else
    {
      TreeType::SingleTreeTraverser traverser(rule);
      for (size_t i = 0; i < y.n_cols; i++)
        traverser.Traverse(i, tree);
    }
    g /= -sumQ;

    // Positive Force Calculation
    for (size_t i = 0; i < y.n_cols; i++)
    {
      for (size_t j = 0; j < N.n_rows; j++)
      {
        const size_t idx = N(j, i);
        const double distanceSq = DistanceType::Evaluate(y.col(i), y.col(idx));

        const double q = 1.0 / (1.0 + distanceSq);
        g.col(i) += q * P(i, idx) * (y.col(i) - y.col(idx));
        error += P(i, idx) *
                 std::log(P(i, idx) / std::max(arma::datum::eps, q / sumQ));
      }
    }

    g *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom;

    return error;
  }

  //! Get the Input Joint Probabilities.
  const arma::sp_mat& InputJointProbabilities() const { return P; }
  //! Modify the Input Joint Probabilities.
  arma::sp_mat& InputJointProbabilities() { return P; }

 private:
  //! Degrees of freedom
  size_t degrees_of_freedom;

  //! Input joint probabilities
  arma::sp_mat P;

  //! Nearest neighbor indexes
  arma::Mat<size_t> N;

  //! Nearest neibhbor distances
  MatType D;

  //! The perplexity of the Gaussian distribution.
  double perplexity;

  //! The coarseness of the approximation.
  double theta;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
