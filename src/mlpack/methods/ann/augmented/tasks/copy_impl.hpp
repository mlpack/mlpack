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

#include <iostream>

using std::cerr;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

CopyTask::CopyTask(int maxLength, int nRepeats) :
    maxLength(maxLength),
    nRepeats(nRepeats)
{
  // Just storing task-specific paramters.
}

template<typename ModelType>
double CopyTask::Evaluate(ModelType& model) {
  int trainSize = 5, testSize = 5;
  int nEpochs = 10;
  arma::field<arma::irowvec> trainInput(trainSize);
  arma::field<arma::irowvec> trainLabels(trainSize);
  arma::field<arma::irowvec> testInput(testSize);
  arma::field<arma::irowvec> testLabels(testSize);
  GenerateData(trainInput, trainLabels, trainSize);
  GenerateData(testInput, testLabels, testSize);
  for (size_t epoch = 0; epoch < nEpochs; epoch++)
  {
    for (size_t el = 0; el < trainSize; el++)
    {
      arma::field<arma::irowvec> inputTemp;
      inputTemp = trainInput.rows(el, el);
      arma::field<arma::irowvec> labelsTemp;
      labelsTemp = trainLabels.rows(el, el);

      model.Train(inputTemp, labelsTemp);
    }
  }
  double score = 0;
  for (size_t i = 0; i < testSize; i++)
  {
    arma::irowvec prediction;
    auto input = testInput.at(i);
    auto output = testLabels.at(i);

    model.Predict(input, prediction);

    if (IsCorrect(output, prediction)) ++score;
  }
  score /= testSize;
  return score;
}

bool CopyTask::IsCorrect(
    arma::irowvec& trueOutput,
    arma::irowvec& predOutput) {
  if (trueOutput.n_elem != predOutput.n_elem) {
    return false;
  }
  if (trueOutput.n_elem != predOutput.n_elem) return false;
  for (int j = 0; j < predOutput.n_elem; ++j) {
    if (trueOutput.at(j) != predOutput.at(j)) return false;
  }
  return true;
}

void CopyTask::GenerateData(
  arma::field<arma::irowvec>& input,
  arma::field<arma::irowvec>& labels,
  int batchSize
) {
  input = arma::field<arma::irowvec>(batchSize);
  labels = arma::field<arma::irowvec>(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    size_t size = maxLength;
    arma::irowvec item = arma::randi<arma::irowvec>(size, arma::distr_param(0, 1));
    input(i) = item;
    arma::irowvec item_ans = arma::irowvec(nRepeats * size);
    for (int r = 0; r < nRepeats; ++r) {
        item_ans.cols(r*size, (r+1)*size-1) = item;
    }
    labels(i) = item_ans;
  }
}

}
}
}
}

#endif