/**
 * @file methods/tsne/tsne_function/tsne_approx_function_impl.hpp
 * @author Ranjodh Singh
 *
 * t-SNE Approx Function Implementation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP

#include "./tsne_approx_function.hpp"
#include <armadillo>
#include <limits>

namespace mlpack {

template <bool UseDualTree, typename DistanceType, typename MatType>
TSNEApproxFunction<UseDualTree, DistanceType, MatType>::TSNEApproxFunction(
    const MatType& X,
    const double perplexity,
    const size_t dof,
    const double theta)
    : perplexity(perplexity), dof(dof), theta(theta)
{
  // To Do: Make number of neighbors a parameter

  // Run KNN
  NeighborSearch<NearestNeighborSort, DistanceType> knn(X);
  const size_t neighbors = std::min<size_t>(
      X.n_cols - 1, static_cast<size_t>(3 * perplexity));
  knn.Search(neighbors, N, D);

  // Square if not SquaredEuclideanDistance
  if (!std::is_same_v<DistanceType, SquaredEuclideanDistance>)
    D = arma::square(D);

  // Precompute P
  P = computeInputJointProbabilities(perplexity, N, D);
}

template <bool UseDualTree, typename DistanceType, typename MatType>
template <typename GradType>
double TSNEApproxFunction<
    UseDualTree,
    DistanceType,
    MatType
>::EvaluateWithGradient(const MatType& y, GradType& g)
{
  // Calculate Negative Gradient
  using tag = typename std::integral_constant<bool, UseDualTree>;
  const double sumQ = CalculateNegativeGradient(g, y, tag{});
  const double error = CalculatePositiveGradient(g, y, sumQ);
  g *= 2.0 * (1.0 + dof) / dof;

  return error;
}

template <bool UseDualTree, typename DistanceType, typename MatType>
double TSNEApproxFunction<
    UseDualTree,
    DistanceType,
    MatType
>::CalculateNegativeGradient(
    MatType& g, const MatType& y, std::true_type /* tag */)
{
  // Init
  double sumQ = 0.0;
  std::vector<size_t> oldFromNew;
  TreeType tree(y, oldFromNew);
  RuleType rule(sumQ, g, y, oldFromNew, dof, theta);
  TreeType::DualTreeTraverser traverser(rule);

  // Traverse
  traverser.Traverse(tree, tree);

  // Normalize
  sumQ = std::max(arma::datum::eps, sumQ);
  g /= -sumQ;

  return sumQ;
}

template <bool UseDualTree, typename DistanceType, typename MatType>
double TSNEApproxFunction<
    UseDualTree,
    DistanceType,
    MatType
>::CalculateNegativeGradient(
    MatType& g, const MatType& y, std::false_type /* tag */)
{
  // To Do: Instead of relying on number of threads for sumQ
  // Make a vector of length equal to number of points.

  // Init
  std::vector<size_t> oldFromNew;
  TreeType tree(y, oldFromNew);
  const size_t maxThreadCount = omp_get_max_threads();
  std::vector<double> localSumQs(maxThreadCount);

  // Traverse
  #pragma omp parallel
  {
    size_t threadId = 0;
    #ifdef MLPACK_USE_OPENMP
      threadId = omp_get_thread_num();
    #endif

    RuleType rule(localSumQs[threadId], g, y, oldFromNew, dof, theta);
    TreeType::SingleTreeTraverser traverser(rule);

    #pragma omp for schedule(static)
    for (size_t i = 0; i < y.n_cols; i++)
      traverser.Traverse(i, tree);
  }

  // Normalize
  const double sumQ = std::max(arma::datum::eps,
      std::accumulate(localSumQs.begin(), localSumQs.end(), 0.0));
  g /= -sumQ;

  return sumQ;
}

template <bool UseDualTree, typename DistanceType, typename MatType>
double TSNEApproxFunction<
    UseDualTree,
    DistanceType,
    MatType
>::CalculatePositiveGradient(MatType& g, const MatType& y, const double sumQ)
{
  // To Do: Instead of relying on number of threads for sumQ
  // Make a vector of length equal to number of points.
  const size_t maxThreadCount = omp_get_max_threads();
  std::vector<double> localErrors(maxThreadCount);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < y.n_cols; i++)
  {
    size_t threadId = 0;
    #ifdef MLPACK_USE_OPENMP
        threadId = omp_get_thread_num();
    #endif

    for (size_t j = 0; j < N.n_rows; j++)
    {
      const size_t idx = N(j, i);
      const double distanceSq = DistanceType::Evaluate(y.col(i), y.col(idx));
      double q = (double)dof / (dof + distanceSq);
      if (dof != 1)
        q = std::pow(q, (1.0 + dof) / 2.0);

      g.col(i) += q * P(i, idx) * (y.col(i) - y.col(idx));
      localErrors[threadId] += P(i, idx) * std::log(
          std::max<double>(arma::datum::eps, P(i, idx)) /
          std::max<double>(arma::datum::eps, q / sumQ));
    }
  }

  return std::accumulate(localErrors.begin(), localErrors.end(), 0.0);
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP
