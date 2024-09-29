/**
 * @file methods/ann/augmented/tasks/sort.hpp
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

/**
 * Generator of instances of the sequence sort task.
 * The parameters are:
 * - maximum sequence length;
 * - binary length of sequence elements.
 *
 * Generated datasets are compliant with mlpack format -
 * every dataset element is shaped as a vector of
 * length (binary length) * (sequence length).
 *
 * Example of generated dataset (sequence length = 3, binary length = 2):
 * - Input sequences: [1,1,0,0,0,1]
 * (three numbers in the sequence are 11, 00, and 01)
 * - Output sequences: [0,0,0,1,1,1]
 * (00, 01, 11 - reordering of the numbers above in the ascending order)
 *
 */
class SortTask
{
 public:
  /**
   * Creates an instance of the sequence sort task.
   *
   * @param maxLength Maximum length of the number sequence.
   * @param bitLen Binary length of sorted numbers.
   * @param addSeparator Flag indicating whether generator
   *                     should emit separating symbol after input sequence.
   */
  SortTask(const size_t maxLength,
           const size_t bitLen,
           bool addSeparator = false);

  /**
   * Generate dataset of a given size.
   *
   * @param input The variable to store input sequences.
   * @param labels The variable to store output sequences.
   * @param batchSize The dataset size.
   * @param fixedLength Flag indicating whether generator
   *                    should emit sequences of pairwise equal length.
   */
  void Generate(arma::field<arma::mat>& input,
                arma::field<arma::mat>& labels,
                      const size_t batchSize,
                      bool fixedLength = false) const;
  /**
   * Generate dataset of a given size and store it in
   * arma::mat object.
   *
   * @param input The variable to store input sequences.
   * @param labels The variable to store output sequences.
   * @param batchSize The dataset size.
   */
  void Generate(arma::mat& input,
                arma::mat& labels,
                const size_t batchSize) const;

 private:
  // Maximum length of the sequence.
  size_t maxLength;
  // Binary length of sorted numbers.
  size_t bitLen;
  // Flag indicating whether generator should produce
  // separator as part of the sequence
  bool addSeparator;
};

} // namespace mlpack

#include "sort_impl.hpp"
#endif
