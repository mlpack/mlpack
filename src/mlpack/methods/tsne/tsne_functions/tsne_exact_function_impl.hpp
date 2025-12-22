/**
 * @file methods/tsne/tsne_functions/tsne_exact_function.hpp
 * @author Ranjodh Singh
 *
 * Implementation of the exact objective function for t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_IMPL_HPP

#include "./tsne_exact_function.hpp"

namespace mlpack {

template <typename MatType, typename DistanceType>
TSNEExactFunction<MatType, DistanceType>::TSNEExactFunction(
    const MatType& X,
    const double perplexity,
    const size_t dof,
    const double /* theta */)
    : perplexity(perplexity), dof(dof)
{
  // To Do: Implement Seperate Evaluate and Gradient too.
  // To Do: Document the fact that if any other metric is
  // given as tparam, it should not be a squared distance type.
  // Also that the given metric will only be used for calculating P.
  // Since, Q is a student's t-dist and it's kernel depends on euclidean.
  // Also find a way to decide at compile time whether metric needs squaring.

  MatType D = PairwiseDistances(X, DistanceType());

  // Square if not SquaredEuclideanDistance
  if (!std::is_same_v<DistanceType, SquaredEuclideanDistance>)
      D = arma::square(D);

  // Precompute P
  P = computeInputProbabilities(perplexity, D);
  P.clamp(arma::Datum<ElemType>::eps, arma::Datum<ElemType>::inf);
}

template <typename MatType, typename DistanceType>
template <typename GradType>
double TSNEExactFunction<MatType, DistanceType>::EvaluateWithGradient(
    const MatType& y, GradType& g)
{
  q = PairwiseDistances(y, SquaredEuclideanDistance());

  q = (q + dof) / dof;
  q = pow(q, -(1.0 + dof) / 2.0);
  q.diag().zeros();

  Q = q / std::max(arma::Datum<ElemType>::eps, accu(q));
  Q.clamp(arma::Datum<ElemType>::eps, arma::Datum<ElemType>::inf);

  M = (P - Q) % q;
  S = sum(M, 1);

  g = (2.0 * (1.0 + dof) / dof) * (y.each_row() % S.t() - y * M);

  // This is way faster than arma::dot
  return accu(P % log(P / Q));
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_IMPL_HPP
