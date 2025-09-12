/**
 * @file methods/tsne/tsne_functions/tsne_exact_function.hpp
 * @author Ranjodh Singh
 *
 * t-SNE Exact Function
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP
#define MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/cv/metrics/facilities.hpp>

#include "../tsne_utils.hpp"

namespace mlpack
{

template <typename MatType = arma::mat>
class TSNEExactFunction
{
 public:
  using DistanceType = SquaredEuclideanDistance;

  TSNEExactFunction(const MatType& X, const double perplexity)
  {
    // Pre Compute P (P_ij's)
    P = binarySearchPerplexity(perplexity,
                               PairwiseDistances(X, DistanceType()));

    P = P + P.t();
    P /= std::max(arma::datum::eps, arma::accu(P));
  }

  //   double Evaluate(const MatType& y);

  //   double Evaluate(const MatType& y, const size_t i, const size_t
  //   batchSize);

  //   void Gradient(const MatType& y, MatType& gradient);

  //   template <typename GradType>
  //   void Gradient(const MatType& y,
  //                 const size_t i,
  //                 GradType& g,
  //                 const size_t batchSize);

  //   double EvaluateWithGradient(const MatType& y, MatType& g);

  template <typename GradType>
  double EvaluateWithGradient(const MatType& y,
                              const size_t /* i */,
                              GradType& g,
                              const size_t /* batchSize */)
  {
    q = PairwiseDistances(y, DistanceType());
    q = 1.0 / (1.0 + q);
    q.diag().zeros();

    Q = q / std::max(arma::datum::eps, arma::accu(q));
    Q.elem(arma::find(Q < arma::datum::eps)).fill(arma::datum::eps);

    M = (P - Q) % q;
    arma::vec S = arma::sum(M, 1);
    g = 4.0 * (y * arma::diagmat(S) - y * M);

    arma::uvec nz = arma::find(P > 0.0);
    const double kl = arma::accu(P.elem(nz) %
                                 arma::log(P.elem(nz) / Q.elem(nz)));
    return kl;
  }

  void Shuffle() { /* Nothing To Do Here */ }

  size_t NumFunctions() { return P.n_cols; }

  const MatType& InputJointProbabilities() const { return P; }
  MatType& InputJointProbabilities() { return P; }

 private:
  MatType P;
  MatType q, Q, M;
};

} // namespace mlpack


#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP
