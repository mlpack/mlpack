/**
 * @file copy_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of CopyTask class
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

double SequencePrecision(arma::field<arma::irowvec> trueOutputs,
                         arma::field<arma::irowvec> predOutputs)
{
  double score = 0;
  auto testSize = trueOutputs.n_elem;
  assert(testSize == predOutputs.n_elem);

  for (size_t i = 0; i < testSize; i++)
  {
    auto prediction = trueOutputs.at(i);
    auto output = predOutputs.at(i);

    bool ok = true;

    if (output.n_elem != prediction.n_elem)
    {
      ok = false;
    }
    else
    {
      for (size_t j = 0; j < prediction.n_elem; ++j) {
        if (output.at(j) != prediction.at(j)) {
          ok = false;
          break;
        }
      }
    }

    if (ok) score++;
  }
  score /= testSize;
  return score;
}

double SequencePrecision(arma::field<arma::imat> trueOutputs,
                         arma::field<arma::imat> predOutputs)
{
  double score = 0;
  auto testSize = trueOutputs.n_elem;
  assert(testSize == predOutputs.n_elem);

  for (size_t i = 0; i < testSize; i++)
  {
    auto prediction = trueOutputs.at(i);
    auto output = predOutputs.at(i);

    bool ok = true;

    if (output.n_rows != prediction.n_rows ||
        output.n_cols != prediction.n_cols)
    {
      ok = false;
    }
    else
    {
      for (size_t j = 0; j < prediction.n_rows; ++j) {
        for (size_t k = 0; k < prediction.n_cols; ++k) {
          if (output.at(j, k) != prediction.at(j, k)) {
            ok = false;
            break;
          }
        }
      }
    }

    if (ok) score++;
  }
  score /= testSize;
  return score;
}

} // namespace scorers 
} // namespace augmented
} // namespace ann
} // namespace mlpack 

#endif
