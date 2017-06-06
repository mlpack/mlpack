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

#include <cstdlib>
#include <ctime>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

CopyTask::CopyTask(size_t maxLength, size_t nRepeats) :
    maxLength(maxLength),
    nRepeats(nRepeats)
{
  assert(maxLength > 1);
  // Just storing task-specific parameters.
}

void CopyTask::GenerateData(arma::field<arma::irowvec>& input,
                            arma::field<arma::irowvec>& labels,
                            size_t batchSize)
{
  input = arma::field<arma::irowvec>(batchSize);
  labels = arma::field<arma::irowvec>(batchSize);
  std::srand(unsigned(std::time(0)));
  for (size_t i = 0; i < batchSize; ++i) {
    // Random uniform length from [2..maxLength]
    size_t size = 2 + std::rand() % (maxLength - 1);
    input(i) = arma::randi<arma::irowvec>(size, arma::distr_param(0, 1));
    arma::irowvec item_ans = arma::irowvec(nRepeats * size);
    for (size_t r = 0; r < nRepeats; ++r) {
        item_ans.cols(r*size, (r+1)*size-1) = input(i);
    }
    labels(i) = item_ans;
  }
}

} // namespace tasks 
} // namespace augmented
} // namespace ann
} // namespace mlpack 

#endif
