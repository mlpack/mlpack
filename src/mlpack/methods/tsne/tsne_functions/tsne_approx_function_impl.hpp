/**
 * @file methods/tsne/tsne_function/tsne_approx_function_impl.hpp
 * @author Ranjodh Singh
 *
 * Implementation of the approximate objective function for t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP

#include "./tsne_approx_function.hpp"

namespace mlpack {

template <bool UseDualTree, typename MatType, typename DistanceType>
TSNEApproxFunction<UseDualTree, MatType, DistanceType>::TSNEApproxFunction(
    const MatType& X,
    const double perplexity,
    const size_t dof,
    const double theta)
    : perplexity(perplexity), dof(dof), theta(theta)
{
  // Run KNN
  NeighborSearch<NearestNeighborSort, DistanceType> knn(X);
  const size_t neighbors = std::min<size_t>(
      X.n_cols - 1, (size_t)(3 * perplexity));
  knn.Search(neighbors, N, D);

  // Square if not SquaredEuclideanDistance
  if (!std::is_same_v<DistanceType, SquaredEuclideanDistance>)
    D = arma::square(D);

  // Precompute P
  P = computeInputProbabilities(perplexity, N, D);
}

template <bool UseDualTree, typename MatType, typename DistanceType>
template <typename GradType>
double TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::EvaluateWithGradient(const MatType& y, GradType& g)
{
  // Calculate Repulsive Part
  const double sumQ = CalculateRepuliveForces(g, y,
      std::bool_constant<UseDualTree>{});

  // Calculate Attractive Part
  const double error = CalculateAttractiveForces(g, y, sumQ);
  
  g *= 2.0 * (1.0 + dof) / dof;

  return error;
}

template <bool UseDualTree, typename MatType, typename DistanceType>
double TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::CalculateRepuliveForces(
    MatType& g, const MatType& y, std::true_type /* tag */)
{
  // Init
  double sumQ = 0.0;
  std::vector<size_t> oldFromNew;
  TreeType tree(y, oldFromNew);
  RuleType rule(sumQ, g, y, oldFromNew, dof, theta);
  typename TreeType::DualTreeTraverser traverser(rule);

  // Traverse
  traverser.Traverse(tree, tree);

  // Normalize
  sumQ = std::max(std::numeric_limits<double>::epsilon(), sumQ);
  g /= -sumQ;

  return sumQ;
}

template <bool UseDualTree, typename MatType, typename DistanceType>
double TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::CalculateRepuliveForces(
    MatType& g, const MatType& y, std::false_type /* tag */)
{
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
    typename TreeType::SingleTreeTraverser traverser(rule);

    #pragma omp for schedule(static)
    for (size_t i = 0; i < y.n_cols; i++)
      traverser.Traverse(i, tree);
  }

  // Normalize
  const double sumQ = std::max(std::numeric_limits<double>::epsilon(),
      std::accumulate(localSumQs.begin(), localSumQs.end(), 0.0));
  g /= -sumQ;

  return sumQ;
}

template <bool UseDualTree, typename MatType, typename DistanceType>
double TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::CalculateAttractiveForces(MatType& g, const MatType& y, const double sumQ)
{
  const size_t k = N.n_rows;
  const size_t n = N.n_cols;
  const size_t maxThreadCount = omp_get_max_threads();
  std::vector<double> localErrors(maxThreadCount);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < y.n_cols; i++)
  {
    size_t threadId = 0;
    #ifdef MLPACK_USE_OPENMP
        threadId = omp_get_thread_num();
    #endif

    for (size_t j = 0; j < k; j++)
    {
      const size_t idx = N[j * k + i];
      const double distanceSq = (double)DistanceType::Evaluate(y.col(i),
                                                               y.col(idx));

      double q = (double)dof / (dof + distanceSq);
      if (dof != 1)
        q = std::pow(q, (1.0 + dof) / 2.0);
      
      const double p = (double)P[i * n + idx];
      if (p)
      {
        g.col(i) += q * p * (y.col(i) - y.col(idx));
        localErrors[threadId] += p * std::log(
            p / std::max(std::numeric_limits<double>::epsilon(), q / sumQ));
      }
      
    }
  }

  return std::accumulate(localErrors.begin(), localErrors.end(), 0.0);
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP
