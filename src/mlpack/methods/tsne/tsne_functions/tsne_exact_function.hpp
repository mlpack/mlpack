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

namespace mlpack {

/**
 * Calculate gradient of the KL-divergence objective, designed for
 * optimization with ensmallen.
 *
 * @tparam MatType The type of Matrix.
 */
template <typename MatType = arma::mat>
class TSNEExactFunction
{
 public:
  // Convenience typedefs.
  using VecType = typename GetColType<MatType>::type;
  // To Do: Make This A Template Parameters
  using DistanceType = SquaredEuclideanDistance;

  /**
   * Constructs the TSNEExactFunction object.
   *
   * @param X The input data. (Din X N)
   * @param perplexity The perplexity of the Gaussian distribution.
   * @param dof The degrees of freedom.
   */
  TSNEExactFunction(const MatType& X,
                    const double perplexity,
                    const size_t dof,
                    const double /* theta */)
      : perplexity(perplexity), dof(dof)
  {
    // Precompute P
    P = binarySearchPerplexity(perplexity,
                               PairwiseDistances(X, DistanceType()));
    P = P + P.t();
    P /= std::max(arma::datum::eps, arma::accu(P));
  }

  /**
   * EvaluateWithGradient for differentiable function optimizers
   * Evaluates the Kullback–Leibler (KL) divergence between input
   * and the embedding and updates gradients accordingly.
   *
   * @param y Current embedding.
   * @param g Variable to store the new gradient.
   */
  template <typename GradType>
  double EvaluateWithGradient(const MatType& y, GradType& g)
  {
    q = PairwiseDistances(y, DistanceType());
    q = (double)dof / (dof + q);
    // Avoid calls to pow for 2 dimensional embeddings
    if (dof != 1)
      q = arma::pow(q, (1.0 + dof) / 2.0);

    Q = q / arma::accu(q);
    Q.clamp(arma::datum::eps, arma::datum::inf);

    M = (P - Q) % q;
    S = arma::sum(M, 1);

    g = y;
    g.each_row() %= S.t();
    g -= y * M;
    g *= 2.0 * (1.0 + dof) / dof;

    // This is faster than using arma::dot
    return arma::accu(P %
        arma::log(arma::clamp(P, arma::datum::eps, arma::datum::inf) / Q));
  }

  //! Get the input joint probabilities.
  const MatType& InputJointProbabilities() const { return P; }
  //! Modify the input joint probabilities.
  MatType& InputJointProbabilities() { return P; }

 private:
  //! Input joint probabilities.
  MatType P;

  //! Output joint probabilities. (Normalized)
  MatType Q;

  //! Output joint probabilities. (Unnormalized)
  MatType q;

  //! Intermediate matrix used in gradient computation.
  MatType M;

  //! Intermediate vector used in gradient computation.
  VecType S;

  //! The perplexity of the Gaussian distribution.
  double perplexity;

  //! Degrees of Freedom.
  size_t dof;
};

} // namespace mlpack

#endif // MLPACK_METHODS_TSNE_TSNE_FUNCTIONS_TSNE_EXACT_FUNCTION_HPP
