/**
 * @file methods/tsne/tsne_utils.hpp
 * @author Ranjodh Singh
 *
 * t-SNE Utility Functions
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_UTILS_HPP
#define MLPACK_METHODS_TSNE_TSNE_UTILS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

namespace mlpack
{

arma::sp_mat binarySearchPerplexity(const double perplexity,
                                    const arma::Mat<size_t>& N,
                                    const arma::mat& D)
{
  const size_t maxSteps = 100;
  const double tolerance = 1e-5;

  const size_t n = D.n_cols;
  const size_t k = D.n_rows;
  const double hDesired = std::log(perplexity);

  arma::sp_mat P(n, n);
  arma::vec beta(n, arma::fill::ones);

  arma::vec Di;
  for (size_t i = 0; i < n; i++)
  {
    double betamin = -arma::datum::inf;
    double betamax = +arma::datum::inf;
    double sumP, sumDP, hDiff, hApprox;

    if (i % 1000 == 0)
      Log::Info << "Computing P-values for points " << i + 1 << " To "
                << std::min(n, i + 1000) << std::endl;

    Di = D.col(i);
    arma::sp_vec Pi(n);

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
      sumP = std::max(sumP, arma::datum::eps);
      hApprox = std::log(sumP) + beta(i) * sumDP / sumP;
      Pi /= sumP;

      hDiff = hApprox - hDesired;
      if (std::abs(hDiff) <= tolerance)
        break;

      if (hDiff > 0)
      {
        betamin = beta(i);
        if (std::isinf(betamax))
          beta(i) *= 2.0;
        else
          beta(i) = (beta(i) + betamax) / 2.0;
      }
      else
      {
        betamax = beta(i);
        if (std::isinf(betamin))
          beta(i) /= 2.0;
        else
          beta(i) = (beta(i) + betamin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    for (size_t j = 0; j < k; ++j)
      P(i, N(j, i)) = Pi(N(j, i));
  }
  Log::Info << "Mean value of sigma: " << arma::mean(arma::sqrt(1.0 / beta))
            << std::endl;

  return P;
}

arma::mat binarySearchPerplexity(const double perplexity, const arma::mat& D)
{
  const size_t maxSteps = 50;
  const double tolerance = 1e-5;

  size_t n = D.n_cols;
  double H = std::log(perplexity);
  arma::vec beta(n, arma::fill::ones);
  arma::mat P(n, n, arma::fill::zeros);

  arma::vec Di, Pi;
  double betamin, betamax;
  double sumP, Happrox, Hdiff;
  for (size_t i = 0; i < n; i++)
  {
    if (i % 1000 == 0)
      Log::Info << "Computing P-values for points " << i + 1 << " To "
                << std::min(n, i + 1000) << std::endl;

    betamin = -arma::datum::inf;
    betamax = +arma::datum::inf;
    Di = D.col(i);
    Di.shed_row(i);

    size_t step = 0;
    while (step < maxSteps)
    {
      Pi = arma::exp(-Di * beta(i));
      sumP = std::max(arma::datum::eps, arma::accu(Pi));
      Happrox = std::log(sumP) + beta(i) * arma::accu(Di % Pi) / sumP;
      Pi /= sumP;

      Hdiff = Happrox - H;
      if (std::abs(Hdiff) <= tolerance)
        break;

      if (Hdiff > 0)
      {
        betamin = beta(i);
        if (std::isinf(betamax))
          beta(i) *= 2.0;
        else
          beta(i) = (beta(i) + betamax) / 2.0;
      }
      else
      {
        betamax = beta(i);
        if (std::isinf(betamin))
          beta(i) /= 2.0;
        else
          beta(i) = (beta(i) + betamin) / 2.0;
      }
      step++;
    }
    if (step == maxSteps)
      Log::Warn << "Max Steps Reached While Searching For Desired Perplexity"
                << std::endl;

    P.row(i).head(i) = Pi.head(i).t();
    P.row(i).tail(n - i - 1) = Pi.tail(n - i - 1).t();
  }
  Log::Info << "Mean value of sigma: " << arma::mean(arma::sqrt(1.0 / beta))
            << std::endl;

  return P;
}

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_UTILS_HPP
