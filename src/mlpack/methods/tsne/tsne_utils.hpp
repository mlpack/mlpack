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
 * Compute the (sparse) input probabilities P for t-SNE from
 * a k-nearest-neighbors distance matrix.
 *
 * This function performs the binary-search-over-sigma procedure described in
 * the original t-SNE paper to obtain conditional probabilities for each
 * data point over its k nearest neighbors.
 *
 * When `normalize` is true, which is the default, the conditional
 * probabilities are symmetrized and normalized to produce a joint distribution
 * given by:
 * \f[
 * P_{ij} = \frac{P_{i|j} + P_{j|i}}{2 \cdot n_{\text{samples}}}
 * \f]
 *
 * @tparam eT Element type for numeric values (usually double or float).
 * @tparam MatType Dense matrix type (defaults to arma::Mat<eT>).
 * @tparam SpMatType Sparse matrix type (defaults to arma::SpMat<eT>).
 * @tparam VecType Dense vector type (defaults to arma::Col<eT>).
 * @tparam SpVecType Sparse vector type (defaults to arma::SpCol<eT>).
 *
 * @param perplexity Desired perplexity.
 * @param N A k x n matrix containing the neighbor indices (column i lists
 *     the indices of the k nearest neighbors of point i in the dataset).
 * @param D A k x n matrix containing distances from each point to its k
 *     nearest neighbors (column i contains distances for neighbors of i).
 * @param normalize If true, conditional probabilities are symmetrized and
 *     normalized into a joint distribution.
 *
 * @return A sparse n x n matrix P containing the required probabilities.
 */
template <typename eT,
          typename MatType = arma::Mat<eT>,
          typename SpMatType = arma::SpMat<eT>,
          typename VecType = arma::Col<eT>,
          typename SpVecType = arma::SpCol<eT>>
SpMatType computeInputProbabilities(const double perplexity,
                                    const arma::Mat<size_t>& N,
                                    const arma::Mat<eT>& D,
                                    bool normalize = true)
{
  const size_t maxSteps = 100;
  const double tolerance = 1e-5;

  const size_t n = D.n_cols;
  const size_t k = D.n_rows;
  const double hDesired = std::log(perplexity);

  SpMatType P(n, n);
  std::vector<double> beta(n, 1.0);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    VecType Di = D.col(i), Pi(k);

    double betaMin, betaMax, sumP, hApprox, hDiff;
    betaMin = -std::numeric_limits<double>::infinity();
    betaMax = +std::numeric_limits<double>::infinity();

    size_t step = 0;
    while (step < maxSteps)
    {
      Pi = exp(-Di * beta[i]);
      sumP = std::max(
          std::numeric_limits<double>::epsilon(), (double)accu(Pi));
      hApprox = std::log(sumP) + beta[i] * (double)accu(Di % Pi) / sumP;
      Pi /= sumP;

      hDiff = hApprox - hDesired;
      if (std::abs(hDiff) <= tolerance)
        break;

      if (hDiff > 0)
      {
        betaMin = beta[i];
        if (std::isinf(betaMax))
          beta[i] *= 2.0;
        else
          beta[i] = (beta[i] + betaMax) / 2.0;
      }
      else
      {
        betaMax = beta[i];
        if (std::isinf(betaMin))
          beta[i] /= 2.0;
        else
          beta[i] = (beta[i] + betaMin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    for (size_t j = 0; j < k; ++j)
      P(i, N(j, i)) = Pi[j];
  }

  // Symmetrize and normalize P
  if (normalize)
    P = (P + P.t()) / (2 * std::max(arma::Datum<eT>::eps, accu(P)));

  return P;
}


/**
 * Compute the (dense) input probability matrix P for t-SNE from a full
 * n x n distance matrix.
 *
 * This function performs the binary-search-over-sigma procedure described in
 * the original t-SNE paper to obtain conditional probabilities for each
 * data point over all the other points.
 *
 * When `normalize` is true, which is the default, the conditional
 * probabilities are symmetrized and normalized to produce a joint distribution
 * given by:
 * \f[
 * P_{ij} = \frac{P_{i|j} + P_{j|i}}{2 \cdot n_{\text{samples}}}
 * \f]
 *
 * This overload performs the binary search on a full n x n distance matrix.
 * If you are using a k-nearest-neighbors approximation, use the sparse
 * overload instead.
 *
 * @tparam eT Element type for numeric values (usually double or float).
 * @tparam MatType Dense matrix type (defaults to arma::Mat<eT>).
 * @tparam VecType Dense vector type (defaults to arma::Col<eT>).
 *
 * @param perplexity Desired perplexity.
 * @param D n x n distance matrix where column i contains distances from
 *     point i to every other point. The diagonal entry D(i,i) is ignored.
 * @param normalize If true, conditional probabilities are symmetrized and
 *     normalized into a joint distribution.
 *
 * @return A dense n x n matrix P containing the required probabilities.
 */
template <typename eT,
          typename MatType = arma::Mat<eT>,
          typename VecType = arma::Col<eT>>
MatType computeInputProbabilities(const double perplexity,
                                  const arma::Mat<eT>& D,
                                  bool normalize = true)
{
  const size_t maxSteps = 100;
  const double tolerance = 1e-5;

  const size_t n = D.n_cols;
  const double hDesired = std::log(perplexity);

  MatType P(n, n, arma::fill::zeros);
  std::vector<double> beta(n, 1.0);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    VecType Di = D.col(i), Pi;
    Di.shed_row(i);

    double betaMin, betaMax, sumP, hApprox, hDiff;
    betaMin = -std::numeric_limits<double>::infinity();
    betaMax = +std::numeric_limits<double>::infinity();

    size_t step = 0;
    while (step < maxSteps)
    {
      Pi = exp(-Di * beta[i]);
      sumP = std::max(
          std::numeric_limits<double>::epsilon(), (double)accu(Pi));
      hApprox = std::log(sumP) + beta[i] * (double)accu(Di % Pi) / sumP;
      Pi /= sumP;

      hDiff = hApprox - hDesired;
      if (std::abs(hDiff) <= tolerance)
        break;

      if (hDiff > 0)
      {
        betaMin = beta[i];
        if (std::isinf(betaMax))
          beta[i] *= 2.0;
        else
          beta[i] = (beta[i] + betaMax) / 2.0;
      }
      else
      {
        betaMax = beta[i];
        if (std::isinf(betaMin))
          beta[i] /= 2.0;
        else
          beta[i] = (beta[i] + betaMin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    P.row(i).head(i) = Pi.head(i).t();
    P.row(i).tail(n - i - 1) = Pi.tail(n - i - 1).t();
  }

  // Symmetrize and normalize P
  if (normalize)
    P = (P + P.t()) / (2 * std::max(arma::Datum<eT>::eps, accu(P)));

  return P;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_UTILS_HPP
