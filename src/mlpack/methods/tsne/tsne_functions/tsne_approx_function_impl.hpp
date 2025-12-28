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

  P = computeInputSimilarities(perplexity, N, D);
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

  sumQ = std::max(DBL_EPSILON, sumQ);
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
  double sumQ = 0.0;
  std::vector<size_t> oldFromNew;
  TreeType tree(y, oldFromNew);

  RuleType rule(sumQ, g, y, oldFromNew, dof, theta);
  typename TreeType::SingleTreeTraverser traverser(rule);

  #pragma omp for schedule(static)
  for (size_t i = 0; i < y.n_cols; i++)
    traverser.Traverse(i, tree);

  sumQ = std::max(DBL_EPSILON, sumQ);
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

  double error = 0.0;
  #pragma omp parallel for reduction(+:error)
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      const size_t idx = N(j, i);
      const double p = (double)P(i, idx);

      const double distanceSq = (double)SquaredEuclideanDistance::Evaluate(
          y.col(i), y.col(idx));

      double q = (double)dof / (dof + distanceSq);
      if (dof != 1)
        q = std::pow(q, (1.0 + dof) / 2.0);

      g.col(i) += p * q * (y.col(i) - y.col(idx));

      error += p * std::log(
          std::max(DBL_EPSILON, p) / std::max(DBL_EPSILON, q / sumQ));
    }
  }

  return error;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_APPROX_FUNCTION_IMPL_HPP
