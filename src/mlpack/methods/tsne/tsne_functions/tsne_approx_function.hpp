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
#include "../centroid_statistic.hpp"
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
 * @tparam MatType The type of Matrix.
 */
template <bool UseDualTree,
          typename DistanceType = SquaredEuclideanDistance,
          typename TreeType = Octree<DistanceType, CentroidStatistic>,
          typename MatType = arma::mat>
class TSNEApproxFunction
{
 public:
  // Convenience typedefs.
  using ElemType = typename MatType::elem_type;
  using SpMatType = typename GetSparseMatType<MatType>::type;
  using RuleType = TSNERules<DistanceType, TreeType, MatType>;

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
                     const double theta = 0.5)
      : perplexity(perplexity), dof(dof), theta(theta)
  {
    // Run KNN
    // To Do: Make number of neibhors a parameter
    NeighborSearch<NearestNeighborSort, DistanceType> knn(X);
    const size_t neighbors = static_cast<size_t>(3 * perplexity + 1);
    knn.Search(neighbors, N, D);

    // Precompute P
    P = binarySearchPerplexity(perplexity, D, N);
    P = P + P.t();
    P /= std::max(arma::datum::eps, arma::accu(P));
  }

  /**
   * EvaluateWithGradient for differentiable function optimizers
   * Evaluates the Kullback–Leibler (KL) divergence between input
   * and the embedding and updates gradients.
   *
   * @param y Current embedding.
   * @param g Variable to store the new gradient.
   */
  template <typename GradType>
  double EvaluateWithGradient(const MatType& y, GradType& g)
  {
    // Init
    double sumQ = 0.0, error = 0.0;
    std::vector<size_t> oldFromNew;

    TreeType tree(y, oldFromNew);
    RuleType rule(sumQ, g, y, oldFromNew, dof, theta);

    // Negative Force Calculation
    if constexpr (UseDualTree)
    {
      typename TreeType::DualTreeTraverser traverser(rule);
      traverser.Traverse(tree, tree);
    }
    else
    {
      typename TreeType::SingleTreeTraverser traverser(rule);
      for (size_t i = 0; i < y.n_cols; i++)
        traverser.Traverse(i, tree);
    }
    sumQ = std::max(arma::datum::eps, sumQ);
    g /= -sumQ;

    // Positive Force Calculation
    for (size_t i = 0; i < y.n_cols; i++)
    {
      for (size_t j = 0; j < N.n_rows; j++)
      {
        const size_t idx = N(j, i);
        const double distanceSq = DistanceType::Evaluate(y.col(i), y.col(idx));

        double q = (double)dof / (dof + distanceSq);
        if (dof != 1)
          q = std::pow<double>(q, (1.0 + dof) / 2.0);

        g.col(i) += q * P(i, idx) * (y.col(i) - y.col(idx));
        error += P(i, idx) *
                 std::log(std::max<double>(arma::datum::eps, P(i, idx)) /
                          std::max<double>(arma::datum::eps, q / sumQ));
      }
    }
    g *= 2.0 * (1.0 + dof) / dof;

    return error;
  }

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

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
