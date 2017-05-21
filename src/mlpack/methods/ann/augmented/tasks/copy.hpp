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
  CopyTask(int maxLength, int nRepeats);

  /**
  * Train the model on the task data and evaluates it.
  * Return average precision of model
  * (number of correctly predicted sequences to
  * total number of sequences)
  *
  * @tparam ModelType The evaluated model type.
  * @param model The model to be evaluated.
  */
  template<typename ModelType>
  double Evaluate(ModelType& model);
private: 
  /**
  * Generate dataset of a given size.
  *
  * @param input The variable to store input sequences.
  * @param labels The variable to store output sequences.
  * @param batchSize The dataset size.
  */
  void GenerateData(
    arma::field<arma::irowvec>& input,
    arma::field<arma::irowvec>& labels,
    int batchSize
  ); 
  /**
  * Function that validates the model's answer against ground truth answer.
  *
  * @param trueOutput Ground truth sequence.
  * @param predOutput Sequence predicted by model.
  */
  bool IsCorrect(arma::irowvec& trueOutput,
                 arma::irowvec& predOutput);
  // Maximum length of a sequence.
  int maxLength;
  // Nomber of repeats the model has to perform to complete the task.
  int nRepeats;
};
}
}
}
}

#include "copy_impl.hpp"

#endif
