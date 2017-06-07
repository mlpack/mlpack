/**
 * @file sort.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the SortTask class, which implements a generator of
 * instances of sequence sort task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_AUGMENTED_TASKS_SORT_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_SORT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {
class SortTask
{
 public:
  SortTask(size_t maxLength, size_t bitLen);

  void GenerateData(arma::field<arma::imat>& input,
                    arma::field<arma::imat>& labels,
                    size_t batchSize);

 private:
  size_t maxLength;
  size_t bitLen;
};
} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "sort_impl.hpp"
#endif



