/**
 * @file methods/ann/augmented/tasks/copy_impl.hpp
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

namespace mlpack {

inline CopyTask::CopyTask(const size_t maxLength,
                          const size_t nRepeats,
                          const bool addSeparator) :
    maxLength(maxLength),
    nRepeats(nRepeats),
    addSeparator(addSeparator)
{
  if (maxLength <= 1)
  {
    std::ostringstream oss;
    oss << "CopyTask::CopyTask(): maximum sequence length ("
        << maxLength << ") "
        << "should be at least 2!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
  if (nRepeats <= 0)
  {
    std::ostringstream oss;
    oss << "CopyTask::CopyTask(): repetition count (" << nRepeats << ") "
        << "is not positive!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
  // Just storing task-specific parameters.
}

inline void CopyTask::Generate(arma::field<arma::mat>& input,
                               arma::field<arma::mat>& labels,
                               const size_t batchSize,
                               bool fixedLength) const
{
  input = arma::field<arma::mat>(batchSize);
  labels = arma::field<arma::mat>(batchSize);
  size_t size = maxLength;
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (!fixedLength)
    {
      arma::vec weights(maxLength - 1);

      DiscreteDistribution<> d(1);
      // We have two binary numbers with exactly two digits (10 and 11).
      // Increasing length by 1 double the number of valid numbers.
      d.Probabilities(0) =
          exp2(arma::linspace(1, maxLength - 1, maxLength - 1));

      size = 2 + d.Random()(0);
    }
    arma::colvec vecInput = randi<arma::colvec>(
      size, DistrParam(0, 1));
    arma::colvec vecLabel = ConvTo<arma::colvec>::From(
        repmat(vecInput, nRepeats, 1));
    size_t totSize = vecInput.n_elem + addSeparator + vecLabel.n_elem;
    input(i) = zeros(totSize, 2);
    input(i).col(0).rows(0, vecInput.n_elem - 1) =
        vecInput;
    if (addSeparator)
      input(i).at(vecInput.n_elem, 0) = 0.5;
    input(i).col(1).rows(addSeparator + vecInput.n_elem, totSize - 1) =
        ones(totSize - vecInput.n_elem - addSeparator);
    input(i) = input(i).t();
    input(i).reshape(input(i).n_elem, 1);
    labels(i) = zeros(totSize, 1);
    labels(i).col(0).rows(addSeparator + vecInput.n_elem, totSize - 1) =
        vecLabel;
  }
}

inline void CopyTask::Generate(arma::mat& input,
                               arma::mat& labels,
                               const size_t batchSize) const
{
  arma::field<arma::mat> fieldInput, fieldLabels;
  Generate(fieldInput, fieldLabels, batchSize, true);
  size_t cols = batchSize;
  input = zeros(fieldInput(0).n_rows, cols);
  labels = zeros(fieldLabels(0).n_rows, cols);
  for (size_t i = 0; i < cols; ++i)
  {
    input.col(i) = fieldInput.at(i);
    labels.col(i) = fieldLabels.at(i);
  }
}

} // namespace mlpack

#endif
