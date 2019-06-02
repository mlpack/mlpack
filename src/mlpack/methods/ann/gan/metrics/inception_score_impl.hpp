/**
 * @file inception_score_impl.hpp
 * @author Saksham Bansal
 *
 * Definition of Inception Score for Generative Adversarial Networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_METRICS_INCEPTION_SCORE_IMPL_HPP
#define MLPACK_METHODS_METRICS_INCEPTION_SCORE_IMPL_HPP

// In case it hasn't been included yet.
#include "inception_score.hpp"

namespace mlpack {
namespace ann /* Artificial Neural Network */ {

template<typename ModelType>
double InceptionScore(ModelType model,
                      arma::mat images)
{
  arma::mat preds;
  model.Predict(images, preds);
  arma::colvec c = arma::log(arma::mean(preds, 1));
  arma::mat temp = arma::log(preds);
  temp.each_col() -= c;
  preds %= temp;

  double score  = arma::as_scalar(arma::mean(arma::sum(preds, 0)));
  return exp(score);
}

} // namespace ann
} // namespace mlpack

#endif
