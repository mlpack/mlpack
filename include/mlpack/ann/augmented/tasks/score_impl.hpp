/**
 * @file methods/ann/augmented/tasks/score_impl.hpp
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

template<typename MatType>
double SequencePrecision(arma::field<MatType> trueOutputs,
                         arma::field<MatType> predOutputs,
                         double tol)
{
  double score = 0;
  size_t testSize = trueOutputs.n_elem;
  if (trueOutputs.n_elem != predOutputs.n_elem)
  {
    std::ostringstream oss;
    oss << "SequencePrecision(): number of predicted sequences ("
        << predOutputs.n_elem << ") should be equal to the number "
        << "of ground-truth sequences ("
        << trueOutputs.n_elem << ")"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }

  for (size_t i = 0; i < testSize; ++i)
  {
    arma::vec delta = vectorise(abs(trueOutputs.at(i) - predOutputs.at(i)));
    double maxDelta = max(delta);
    if (maxDelta < tol)
    {
      score++;
    }
  }
  score /= testSize;
  return score;
}

} // namespace mlpack

#endif
