/**
 * @file copy.hpp
 * @author Konstantin Sidorov
 *
 * Definition of the CopyTask class, which implements a generator of
 * instances of sequence copy task.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_AUGMENTED_TASKS_COPY_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_COPY_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {
class CopyTask
{
public:
  /**
  * Creates an instance of the sequence copy task.
  *
  * @param maxLength Maximum length of sequence that has to be repeated by model.
  * @param nRepeats Number of repeates required to solve the task.
  */
  CopyTask(size_t maxLength, size_t nRepeats);
  /**
  * Generate dataset of a given size.
  *
  * @param input The variable to store input sequences.
  * @param labels The variable to store output sequences.
  * @param batchSize The dataset size.
  */
  void GenerateData(arma::field<arma::irowvec>& input,
                    arma::field<arma::irowvec>& labels,
                    size_t batchSize);
private:
  // Maximum length of a sequence.
  size_t maxLength;
  // Number of repeats the model has to perform to complete the task.
  size_t nRepeats;
};
} // namespace tasks 
} // namespace augmented
} // namespace ann
} // namespace mlpack 

#include "copy_impl.hpp"

#endif
