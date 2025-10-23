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

#include <limits>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

namespace mlpack {

/**
 * Compute the (sparse) input joint probabilities P for t-SNE from
 * a k-nearest-neighbors distance matrix.
 *
 * This function performs the binary-search-over-sigma procedure described in
 * the original t-SNE paper to obtain conditional probabilities for each
 * datapoint over its k nearest neighbors, It then symmetrizes and normalizes
 * these conditional probabilities into a joint distribution using the formula
 * P_{ij} = (P_{j|i} + P_{i|j}) / (2 * n_samples).
 *
 * @tparam eT Element type for numeric values (usually double or float).
 * @tparam MatType Dense matrix type (defaults to arma::Mat<eT>).
 * @tparam SpMatType Sparse matrix type (defaults to arma::SpMat<eT>).
 * @tparam VecType Dense vector type (defaults to arma::Col<eT>).
 * @tparam SpVecType Sparse vector type (defaults to arma::SpCol<eT>).
 *
 * @param perplexity Desired perplexity (controls the bandwidth search).
 * @param N A k x n matrix containing the neighbor indices (row i lists
 *     the indices of the k nearest neighbors of point i in the dataset).
 * @param D A k x n matrix containing distances from each point to its k
 *     nearest neighbors (row i contains distances for neighbors of i).
 *
 * @return A sparse n x n matrix P containing the joint probabilities.
 */
template <typename eT,
          typename MatType = arma::Mat<eT>,
          typename SpMatType = arma::SpMat<eT>,
          typename VecType = arma::Col<eT>,
          typename SpVecType = arma::SpCol<eT>>
SpMatType computeInputJointProbabilities(const double perplexity,
                                         const arma::Mat<size_t>& N,
                                         const arma::Mat<eT>& D)
{
  const size_t maxSteps = 100;
  const double tolerance = 1e-5;

  const size_t n = D.n_cols;
  const size_t k = D.n_rows;
  const double hDesired = std::log(perplexity);

  SpMatType P(n, n);
  VecType beta(n, arma::fill::ones);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    VecType Di;
    SpVecType Pi(n);
    double betaMin = -arma::datum::inf;
    double betaMax = +arma::datum::inf;
    double sumP, sumDP, hDiff, hApprox;

    Di = D.col(i);
    Pi.zeros();

    size_t step = 0;
    while (step < maxSteps)
    {
      sumP = sumDP = 0.0;
      for (size_t j = 0; j < k; j++)
      {
        Pi(N(j, i)) = std::exp(-Di(j) * beta(i));
        sumP += Pi(N(j, i));
        sumDP += Di(j) * Pi(N(j, i));
      }
      sumP = std::max(arma::datum::eps, sumP);
      hApprox = std::log(sumP) + beta(i) * sumDP / sumP;
      Pi /= sumP;

      hDiff = hApprox - hDesired;
      if (std::abs(hDiff) <= tolerance)
        break;

      if (hDiff > 0)
      {
        betaMin = beta(i);
        if (std::isinf(betaMax))
          beta(i) *= 2.0;
        else
          beta(i) = (beta(i) + betaMax) / 2.0;
      }
      else
      {
        betaMax = beta(i);
        if (std::isinf(betaMin))
          beta(i) /= 2.0;
        else
          beta(i) = (beta(i) + betaMin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    for (size_t j = 0; j < k; ++j)
      P(i, N(j, i)) = Pi(N(j, i));
  }
  Log::Info << "Mean value of sigma: " << std::sqrt(n / arma::accu(beta))
            << std::endl;

  // Symmetrize and Normalize P
  P = P + P.t();
  // Probabilies already sum to n, after transposed addition they sum to 2n.
  P /= std::max(std::numeric_limits<eT>::epsilon(), arma::accu(P));

  return P;
}


/**
 * Compute the (dense) input joint probability matrix P for t-SNE from a full
 * n x n distance matrix.
 *
 * This function performs the binary-search-over-sigma procedure described in
 * the original t-SNE paper to obtain conditional probabilities for each
 * datapoint over all the other points, It then symmetrizes and normalizes
 * these conditional probabilities into a joint distribution using the formula
 * \f[
 * P_{ij} = \frac{P_{j|i} + P_{i|j}}{2 \cdot n_{\text{samples}}}
 * \f]
 *
 * This overload performs the entire per-point binary search directly and is
 * intended for use when a full distance matrix is available. If a k-NN
 * distance representation is used elsewhere, prefer the k-NN (sparse)
 * overload to avoid redundant computation.
 *
 * @tparam eT Element type for numeric values (usually double or float).
 * @tparam MatType Dense matrix type (defaults to arma::Mat<eT>).
 * @tparam VecType Dense vector type (defaults to arma::Col<eT>).
 *
 * @param perplexity Desired perplexity value used to set the effective
 *     neighborhood size for each point.
 * @param D n x n distance matrix where column i contains distances from
 *     point i to every other point. The diagonal entry D(i,i) is ignored.
 *
 * @return A dense n x n matrix P containing the joint probabilities.
 */
template <typename eT,
          typename MatType = arma::Mat<eT>,
          typename VecType = arma::Col<eT>>
MatType computeInputJointProbabilities(const double perplexity,
                                       const arma::Mat<eT>& D)
{
  const size_t maxSteps = 50;
  const double tolerance = 1e-5;

  const size_t n = D.n_cols;
  const double hDesired = std::log(perplexity);

  VecType beta(n, arma::fill::ones);
  MatType P(n, n, arma::fill::zeros);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++)
  {
    VecType Di, Pi;
    double betaMin, betaMax;
    double sumP, hApprox, hDiff;

    betaMin = -arma::datum::inf;
    betaMax = +arma::datum::inf;
    Di = D.col(i);
    Di.shed_row(i);

    size_t step = 0;
    while (step < maxSteps)
    {
      Pi = arma::exp(-Di * beta(i));
      sumP = std::max(arma::datum::eps, arma::accu(Pi));
      hApprox = std::log(sumP) + beta(i) * arma::accu(Di % Pi) / sumP;
      Pi /= sumP;

      hDiff = hApprox - hDesired;
      if (std::abs(hDiff) <= tolerance)
        break;

      if (hDiff > 0)
      {
        betaMin = beta(i);
        if (std::isinf(betaMax))
          beta(i) *= 2.0;
        else
          beta(i) = (beta(i) + betaMax) / 2.0;
      }
      else
      {
        betaMax = beta(i);
        if (std::isinf(betaMin))
          beta(i) /= 2.0;
        else
          beta(i) = (beta(i) + betaMin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    P.row(i).head(i) = Pi.head(i).t();
    P.row(i).tail(n - i - 1) = Pi.tail(n - i - 1).t();
  }
  Log::Info << "Mean value of sigma: " << std::sqrt(n / arma::accu(beta))
            << std::endl;

  // Symmetrize and Normalize P
  P = P + P.t();
  // Probabilies already sum to n, after transposed addition they sum to 2n.
  P /= std::max(std::numeric_limits<eT>::epsilon(), arma::accu(P));

  return P;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_UTILS_HPP
