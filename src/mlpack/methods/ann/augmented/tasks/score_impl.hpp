/**
 * @file score_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of scoring functions
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AUGMENTED_TASKS_SCORE_IMPL_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_SCORE_IMPL_HPP

// In case it hasn't been included yet.
#include "score.hpp"

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace scorers /* Scoring utilities for augmented */ {

template<typename MatType>
double SequencePrecision(arma::field<MatType> trueOutputs,
                         arma::field<MatType> predOutputs)
{
  double score = 0;
  size_t testSize = trueOutputs.n_elem;
  assert(testSize == predOutputs.n_elem);

  for (size_t i = 0; i < testSize; i++)
  {
    MatType delta = arma::abs(trueOutputs.at(i) - predOutputs.at(i));
    delta.reshape(delta.n_elem, 1);
    arma::vec deltaVec = arma::conv_to<arma::vec>::from(delta);
    double maxDelta = arma::max(deltaVec);
    double eps = 1e-4;
    if (maxDelta < eps)
    {
      score++;
    }
  }
  score /= testSize;
  return score;
}

} // namespace scorers
} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif
