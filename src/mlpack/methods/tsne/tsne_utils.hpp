/**
 * @file methods/tsne/tsne_utils.hpp
 * @author Ranjodh Singh
 *
 * t-SNE Utility Functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_UTILS_HPP
#define MLPACK_METHODS_TSNE_TSNE_UTILS_HPP

#include <mlpack/core/util/io.hpp>

namespace mlpack {

/**
 * Perform a binary search to find a conditional
 * distribution with the desired perplexity for a given point.
 *
 * @param DSqi Vector containing squared distances
 *     from the target point to its neighbors.
 * @param perplexity Desired perplexity.
 * @param tolerance Margin of error for the entropy match.
 * @param maxSteps Maximum iterations for the binary search.
 */
template<typename VecType>
inline VecType binarySearchPerplexity(
    const VecType& DSqi,
    const double perplexity,
    const double tolerance = 1e-5,
    const size_t maxSteps = 100)
{
  VecType Pi(size(DSqi));
  const double hDesired = std::log(perplexity);

  double beta, betaMin, betaMax, sumP, hApprox, hDiff;
  betaMin = -std::numeric_limits<double>::infinity();
  betaMax = +std::numeric_limits<double>::infinity();

  beta = 1.0;
  size_t step = 0;
  while (step < maxSteps)
  {
    Pi = exp(-beta * DSqi);
    sumP = std::max(DBL_EPSILON, (double)accu(Pi));
    hApprox = std::log(sumP) + beta * (double)accu(DSqi % Pi) / sumP;
    Pi /= sumP;

    hDiff = hApprox - hDesired;
    if (std::abs(hDiff) <= tolerance)
      return Pi;

    if (hDiff > 0)
    {
      betaMin = beta;
      if (std::isinf(betaMax))
        beta *= 2.0;
      else
        beta = (beta + betaMax) / 2.0;
    }
    else
    {
      betaMax = beta;
      if (std::isinf(betaMin))
        beta /= 2.0;
      else
        beta = (beta + betaMin) / 2.0;
    }
    step++;
  }
  Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
            << std::endl;

  return Pi;
}

/**
 * Computes the n x n probability matrix P (dense) measuring pairwise
 * similarities between points in the original high-dimensional space.
 *
 * @param perplexity Desired perplexity.
 * @param DSq n x n squared distance matrix where column i contains distances
 *     from point i to every other point. The diagonal entries are ignored.
 * @param normalize If true, conditional probabilities are symmetrized and
 *     normalized into a joint probability distribution.
 *
 * @return A dense n x n matrix P containing the required probabilities.
 */
template <typename MatType,
          typename ElemType = typename MatType::elem_type,
          typename VecType = typename GetColType<MatType>::type>
MatType computeInputSimilarities(const double perplexity,
                                 const MatType& DSq,
                                 bool normalize = true)
{
  const size_t n = DSq.n_cols;

  MatType P(n, n);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    VecType DSqi(n - 1);
    DSqi.head(i) = DSq.col(i).head(i);
    DSqi.tail(n - i - 1) = DSq.col(i).tail(n - i - 1);

    VecType p = binarySearchPerplexity(DSqi, perplexity);

    P(i, i) = 0;
    P.col(i).head(i) = p.head(i);
    P.col(i).tail(n - i - 1) = p.tail(n - i - 1);
  }

  if (normalize)
    P = (P + P.t()) / (2 * std::max(arma::Datum<ElemType>::eps, accu(P)));

  return P;
}

/**
 * Computes the n x n probability matrix P (sparse) measuring pairwise
 * similarities between points in the original high-dimensional space.
 *
 * @param perplexity Desired perplexity.
 * @param N A k x n matrix containing the neighbor indices (column i lists
 *     the indices of the k nearest neighbors of point i in the dataset).
 * @param DSq A k x n matrix containing squared distances from each point to
 *     its k nearest neighbors (column i contains distances for neighbors of i).
 * @param normalize If true, conditional probabilities are symmetrized and
 *     normalized into a joint probability distribution.
 *
 * @return A sparse n x n matrix P containing the required probabilities.
 */
template <typename MatType,
          typename UMatType,
          typename ElemType = typename MatType::elem_type,
          typename VecType = typename GetColType<MatType>::type,
          typename SpMatType = typename GetSparseMatType<MatType>::type>
SpMatType computeInputSimilarities(const double perplexity,
                                   const UMatType& N,
                                   const MatType& DSq,
                                   bool normalize = true)
{
  const size_t k = DSq.n_rows;
  const size_t n = DSq.n_cols;

  MatType Ptmp(k, n);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    const VecType& DSqi = DSq.col(i);
    Ptmp.col(i) = binarySearchPerplexity(DSqi, perplexity);
  }

  UMatType locations(2, k * n);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      locations(0, i * k + j) = N(j, i);
      locations(1, i * k + j) = i;
    }
  }
  const VecType &values = Ptmp.as_col();
  SpMatType P(locations, values, n, n);

  if (normalize)
    P = (P + P.t()) / (2 * std::max(arma::Datum<ElemType>::eps, accu(P)));

  return P;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_UTILS_HPP
