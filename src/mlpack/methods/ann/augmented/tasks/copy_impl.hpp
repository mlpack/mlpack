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
#ifndef MLPACK_METHODS_AUGMENTED_TASKS_COPY_IMPL_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_COPY_IMPL_HPP

// In case it hasn't been included yet.
#include "copy.hpp"

using mlpack::math::RandInt;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

CopyTask::CopyTask(const size_t maxLength, const size_t nRepeats) :
    maxLength(maxLength),
    nRepeats(nRepeats)
{
  assert(maxLength > 1);
  // Just storing task-specific parameters.
}

void CopyTask::Generate(arma::field<arma::colvec>& input,
                        arma::field<arma::colvec>& labels,
                        const size_t batchSize)
{
  input = arma::field<arma::colvec>(batchSize);
  labels = arma::field<arma::colvec>(batchSize);
  for (size_t i = 0; i < batchSize; ++i) {
    // Random uniform length from [2..maxLength]
    size_t size = RandInt(2, maxLength+1);
    input(i) = arma::randi<arma::colvec>(size, arma::distr_param(0, 1));
    labels(i) = arma::conv_to<arma::colvec>::from(
      arma::repmat(input(i), nRepeats, 1));
  }
}

} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif
