/**
 * @file methods/ann/gan/metrics/inception_score_impl.hpp
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

template <typename ModelType>
double InceptionScore(ModelType model,
                      arma::mat images,
                      size_t splits)
{
  size_t samples = images.n_cols;
  size_t splitSize = samples / splits;
  size_t remainder = samples % splits;
  arma::mat preds;
  model.Predict(images, preds);

  size_t index = 0;
  arma::vec scores = arma::vec(splits);

  for (size_t i = 0; i < splits; ++i)
  {
    size_t curSize = splitSize;
    if (remainder)
    {
      curSize++;
      remainder--;
    }
    arma::mat curPreds =
        arma::mat(preds.colptr(index), preds.n_rows, curSize, false, true);
    arma::colvec c = log(arma::mean(curPreds, 1));
    arma::mat temp = log(curPreds);
    temp.each_col() -= c;
    curPreds %= temp;
    scores(i) = exp(arma::as_scalar(arma::mean(sum(curPreds, 0))));
    index += curSize;
  }

  return arma::mean(scores);
}

} // namespace mlpack

#endif
