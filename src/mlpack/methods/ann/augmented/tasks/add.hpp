/**
 * @file add.hpp
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

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {
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
  */
  void Generate(arma::field<arma::colvec>& input,
                arma::field<arma::colvec>& labels,
                const size_t batchSize);

 private:
  // Maximum binary length of numbers.
  size_t bitLen;
};
} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack

#include "add_impl.hpp"
#endif



