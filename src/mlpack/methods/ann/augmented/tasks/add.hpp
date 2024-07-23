/**
 * @file methods/ann/augmented/tasks/add.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the AddTask class, which implements a generator of
 * instances of binary addition task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AUGMENTED_TASKS_ADD_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_ADD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distributions/discrete_distribution.hpp>

namespace mlpack {

/**
 * Generator of instances of the binary addition task.
 * The parameters are:
 * - macimum binary length;
 *
 * Every element of sequence is encoded as 1-dimensional vector
 * (possible vector elements are {0, 1, 0.5} -
 * the latter corresponds to '+' sign').
 * Generated datasets are compliant with mlpack format -
 * every dataset element is shaped as a vector of
 * length 3 * (sequence length),
 *
 * Example of generated dataset (binary length = 2):
 * - Input sequence: [0,1,0,0,0,1,0,1,0,1,0,0]
 * - Output sequences: [0,1,0,0,1,0]
 *
 */
class AddTask
{
 public:
  /**
   * Creates an instance of the binary addition task.
   *
   * @param bitLen Maximum binary length of added numbers.
   */
  AddTask(const size_t bitLen);

  /**
   * Generate dataset of a given size.
   *
   * @param input The variable to store input sequences.
   * @param labels The variable to store output sequences.
   * @param batchSize The dataset size.
   * @param fixedLength Flag that indicates whether
   *                    the method should return sequences of even length.
   */
  void Generate(arma::field<arma::mat>& input,
                arma::field<arma::mat>& labels,
                const size_t batchSize,
                const bool fixedLength = false) const;

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
  // Maximum binary length of numbers.
  size_t bitLen;

  /**
   * Function for converting intermediate sequence representation
   * to one-hot representation produced
   * on the final stage of Generate function.
   *
   * @param input Reference parameter with intermediate representations,
   *              in which 0 and 1 corrrespond to the corresponding number bits,
   *              and 2 corresponds to `+` sign, which acts as a separator.
   * @param output Reference parameter for storing final representations.
   */
  void Binarize(const arma::field<arma::vec>& input,
                arma::field<arma::mat>& output) const;
};

} // namespace mlpack

#include "add_impl.hpp"
#endif
