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
#include "../tsne_methods.hpp"
#include "../tsne_approx_rules.hpp"
#include "../centroid_statistic.hpp"

namespace mlpack
{

template <typename TSNEStrategy, typename MatType = arma::mat>
class TSNEApproxFunction
{
 public:
  // To Do
  using DistanceType = SquaredEuclideanDistance;
  using TreeType = Octree<DistanceType, CentroidStatistic>;

  TSNEApproxFunction(const MatType& X,
                     const double perplexity,
                     const double theta = 0.5)
      : theta(theta)
  {
    // Run KNN
    NeighborSearch<NearestNeighborSort, DistanceType> knn(X);
    const size_t neighbors = static_cast<size_t>(3 * perplexity);
    knn.Search(neighbors, N, D);

    // Pre Compute P (P_ij's)
    P = binarySearchPerplexity(perplexity, N, D);
    P = P + P.t();
    P /= std::max(arma::datum::eps, arma::accu(P));
  }

  //   double Evaluate(const MatType& y);

  //   double Evaluate(const MatType& y, const size_t i, const size_t
  //   batchSize);

  //   void Gradient(const MatType& y, MatType& gradient);

  //   template <typename GradType>
  //   void Gradient(const MatType& y,
  //                 const size_t i,
  //                 GradType& g,
  //                 const size_t batchSize);

  //   double EvaluateWithGradient(const MatType& y, MatType& g);

  template <typename GradType>
  double EvaluateWithGradient(const MatType& y,
                              const size_t /* i */,
                              GradType& g,
                              const size_t /* batchSize */)
  {
    double sumQ = 0.0, error = 0.0;
    std::vector<size_t> oldFromNew;
    TreeType tree(y, oldFromNew, 1);
    TSNEApproxRules<TSNEStrategy> rule(sumQ, g, y, oldFromNew, theta);

    if constexpr (std::is_same_v<TSNEStrategy, DualTreeTSNE>)
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
    g = -g / sumQ;

    for (size_t i = 0; i < y.n_cols; i++)
    {
      for (size_t j = 0; j < N.n_rows; j++)
      {
        size_t idx = N(j, i);
        const double distanceSq = DistanceType::Evaluate(y.col(i), y.col(idx));

        const double q = 1.0 / (1.0 + distanceSq);
        g.col(i) += q * P(i, idx) * (y.col(i) - y.col(idx));
        error += P(i, idx) *
                 std::log(P(i, idx) / std::max(arma::datum::eps, q / sumQ));
      }
    }

    g *= 4;
    return error;
  }

  void Shuffle() {}

  size_t NumFunctions() { return P.n_cols; }

  const arma::sp_mat& InputJointProbabilities() const { return P; }
  arma::sp_mat& InputJointProbabilities() { return P; }

 private:
  arma::sp_mat P;
  MatType D;
  arma::Mat<size_t> N;
  const double theta;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_HPP
