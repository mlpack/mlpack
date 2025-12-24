/**
 * @file methods/tsne/tsne_functions/tsne_approx_function_impl.hpp
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
  const size_t neighbors = std::min<size_t>(
      X.n_cols - 1, (size_t)(3.0 * perplexity + 1.0));

  NeighborSearch<NearestNeighborSort, DistanceType, MatType> knn(X);
  knn.Search(neighbors, N, D);
  if (!std::is_same_v<DistanceType, SquaredEuclideanDistance>)
    D = square(D);

  P = computeInputProbabilities(perplexity, N, D);
}

template <bool UseDualTree, typename MatType, typename DistanceType>
template <typename GradType>
typename MatType::elem_type TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::EvaluateWithGradient(const MatType& y, GradType& g)
{
  // Calculate Repulsive Part
  const double sumQ = CalculateRepulsiveForces(g, y,
      std::bool_constant<UseDualTree>{});

  // Calculate Attractive Part
  const ElemType error = CalculateAttractiveForces(g, y, sumQ);

  g *= 2.0 * (1.0 + dof) / dof;

  return error;
}

template <bool UseDualTree, typename MatType, typename DistanceType>
double TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::CalculateRepulsiveForces(
    MatType& g, const MatType& y, std::true_type /* tag */)
{
  double sumQ = 0.0;
  std::vector<size_t> oldFromNew;
  TreeType tree(y, oldFromNew);

  RuleType rule(sumQ, g, y, oldFromNew, dof, theta);
  typename TreeType::DualTreeTraverser traverser(rule);

  traverser.Traverse(tree, tree);

  sumQ = std::max(std::numeric_limits<double>::epsilon(), sumQ);
  g /= -sumQ;

  return sumQ;
}

template <bool UseDualTree, typename MatType, typename DistanceType>
double TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::CalculateRepulsiveForces(
    MatType& g, const MatType& y, std::false_type /* tag */)
{
  std::vector<size_t> oldFromNew;
  TreeType tree(y, oldFromNew);

  size_t maxThreadCount = 0;
  #ifdef MLPACK_USE_OPENMP
    maxThreadCount = omp_get_max_threads();
  #endif
  std::vector<double> localSumQs(maxThreadCount);

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

  const double sumQ = std::max(std::numeric_limits<double>::epsilon(),
      std::accumulate(localSumQs.begin(), localSumQs.end(), 0.0));
  g /= -sumQ;

  return sumQ;
}

template <bool UseDualTree, typename MatType, typename DistanceType>
typename MatType::elem_type TSNEApproxFunction<
    UseDualTree,
    MatType,
    DistanceType
>::CalculateAttractiveForces(MatType& g, const MatType& y, const double sumQ)
{
  const size_t k = N.n_rows;
  const size_t n = N.n_cols;

  size_t maxThreadCount = 0;
  #ifdef MLPACK_USE_OPENMP
    maxThreadCount = omp_get_max_threads();
  #endif
  std::vector<double> localErrors(maxThreadCount);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    size_t threadId = 0;
    #ifdef MLPACK_USE_OPENMP
        threadId = omp_get_thread_num();
    #endif

    for (size_t j = 0; j < k; j++)
    {
      const size_t idx = N(j, i);
      const double distanceSq = (double)SquaredEuclideanDistance::Evaluate(
          y.col(i), y.col(idx));

      double q = (double)dof / (dof + distanceSq);
      if (dof != 1)
        q = std::pow(q, (1.0 + dof) / 2.0);

      const double p = (double)P(i, idx);
      g.col(i) += q * p * (y.col(i) - y.col(idx));

      localErrors[threadId] += p * std::log(
          std::max(std::numeric_limits<double>::epsilon(), p)
          / std::max(std::numeric_limits<double>::epsilon(), q / sumQ));
    }
  }

  return (ElemType)std::accumulate(
      localErrors.begin(), localErrors.end(), 0.0);
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP
