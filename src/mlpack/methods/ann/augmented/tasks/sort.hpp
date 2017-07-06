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
  /**
  * Creates an instance of the sequence sort task.
  *
  * @param maxLength Maximum length of the number sequence.
  * @param bitLen Binary length of sorted numbers.
  */
  SortTask(const size_t maxLength, const size_t bitLen,
           bool addSeparator = false);
  /**
  * Generate dataset of a given size.
  *
  * @param input The variable to store input sequences.
  * @param labels The variable to store output sequences.
  * @param batchSize The dataset size.
  */
  void Generate(arma::field<arma::mat>& input,
                arma::field<arma::mat>& labels,
                const size_t batchSize, 
                bool fixedLength = false);

  void Generate(arma::mat& input,
                arma::mat& labels,
                const size_t batchSize);

 private:
  // Maximum length of the sequence.
  size_t maxLength;
  // Binary length of sorted numbers.
  size_t bitLen;
  // Flag indicating whether generator should produce
  // separator as part of the sequence
  bool addSeparator;
};
} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "sort_impl.hpp"
#endif



