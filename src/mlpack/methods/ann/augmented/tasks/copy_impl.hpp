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

CopyTask::CopyTask(const size_t maxLength, const size_t nRepeats,
                   bool addSeparator = false) :
    maxLength(maxLength),
    nRepeats(nRepeats),
    addSeparator(addSeparator)
{
  assert(maxLength > 1);
  assert(nRepeats > 0);
  // Just storing task-specific parameters.
}

void CopyTask::Generate(arma::field<arma::mat>& input,
                        arma::field<arma::mat>& labels,
                        const size_t batchSize,
                        bool fixedLength = false)
{
  input = arma::field<arma::mat>(batchSize);
  labels = arma::field<arma::mat>(batchSize);
  size_t size = maxLength;
  for (size_t i = 0; i < batchSize; ++i) {
    if (!fixedLength) {
      // Random uniform length from [2..maxLength]
      size = RandInt(2, maxLength+1);
    }
    arma::colvec vecInput = arma::randi<arma::colvec>(
      size, arma::distr_param(0, 1));
    arma::colvec vecLabel = arma::conv_to<arma::colvec>::from(
      arma::repmat(vecInput, nRepeats, 1));
    size_t totSize = vecInput.n_elem + addSeparator + vecLabel.n_elem;
    input(i) = arma::zeros(totSize, 2);
    input(i).col(0).rows(0, vecInput.n_elem-1) =
      vecInput;
    if (addSeparator)
      input(i).at(vecInput.n_elem, 0) = 0.5;
    input(i).col(1).rows(addSeparator+vecInput.n_elem, totSize-1) =
      arma::ones(totSize-vecInput.n_elem-addSeparator);
    input(i) = input(i).t();
    input(i).reshape(input(i).n_elem, 1);
    labels(i) = arma::zeros(totSize, 1);
    labels(i).col(0).rows(addSeparator+vecInput.n_elem, totSize-1) =
      vecLabel;
  }
}

void CopyTask::Generate(arma::mat& input, arma::mat& labels,
                        const size_t batchSize) {
  std::cerr << "Generating\n";
  arma::field<arma::mat> fieldInput, fieldLabels;
  Generate(fieldInput, fieldLabels, batchSize, true);
  size_t input_rows = fieldInput(0).n_rows;
  size_t label_rows = fieldLabels(0).n_rows;
  size_t cols = batchSize;
  input = arma::zeros(input_rows, cols);
  labels = arma::zeros(label_rows, cols);
  for (size_t i = 0; i < cols; ++i) {
    input.col(i) = fieldInput.at(i);
    labels.col(i) = fieldLabels.at(i);
  }
}


} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack

#endif
