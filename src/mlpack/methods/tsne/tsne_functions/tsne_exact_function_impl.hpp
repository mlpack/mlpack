/**
 * @file methods/tsne/tsne_functions/tsne_exact_function_impl.hpp
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
  MatType D = PairwiseDistances(X, DistanceType());
  if (!std::is_same_v<DistanceType, SquaredEuclideanDistance>)
      D = square(D);

  P = computeInputSimilarities(perplexity, D);
  P.clamp(arma::Datum<ElemType>::eps, arma::Datum<ElemType>::inf);
}

template <typename MatType, typename DistanceType>
template <typename GradType>
typename MatType::elem_type TSNEExactFunction<
    MatType,
    DistanceType
>::EvaluateWithGradient(const MatType& y, GradType& g)
{
  q = PairwiseDistances(y, SquaredEuclideanDistance());
  q = pow((q + dof) / dof, -(1.0 + dof) / 2.0);
  q.diag().zeros();

  Q = q / std::max(arma::Datum<ElemType>::eps, accu(q));
  Q.clamp(arma::Datum<ElemType>::eps, arma::Datum<ElemType>::inf);

  deltaPQ = (P - Q) % q;

  const double c = 2.0 * (1.0 + dof) / dof;
  g = c * (y.each_row() % sum(deltaPQ, 1).t() - y * deltaPQ);

  // This is way faster than dot
  return accu(P % log(P / Q));
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_IMPL_HPP
