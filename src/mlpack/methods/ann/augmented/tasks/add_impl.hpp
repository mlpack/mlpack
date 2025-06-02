/**
 * @file methods/ann/augmented/tasks/add_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of AddTask class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_AUGMENTED_TASKS_ADD_IMPL_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_ADD_IMPL_HPP

#include "add.hpp"

namespace mlpack {

inline AddTask::AddTask(const size_t bitLen) : bitLen(bitLen)
{
  if (bitLen <= 0)
  {
    std::ostringstream oss;
    oss << "AddTask::AddTask(): binary length (" << bitLen << ") "
        << "is not positive!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
}

inline void AddTask::Generate(arma::field<arma::mat>& input,
                              arma::field<arma::mat>& labels,
                              const size_t batchSize,
                              bool fixedLength) const
{
  arma::field<arma::vec> vecInput = arma::field<arma::colvec>(batchSize);
  arma::field<arma::vec> vecLabels = arma::field<arma::colvec>(batchSize);
  size_t sizeA = bitLen, sizeB = bitLen;
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (!fixedLength)
    {
      arma::vec weights(bitLen - 1);
      weights = exp2(arma::linspace(1, bitLen - 1, bitLen - 1));

      DiscreteDistribution<> d(1);
      // We have two binary numbers with exactly two digits (10 and 11).
      // Increasing length by 1 double the number of valid numbers.
      d.Probabilities(0) = exp2(arma::linspace(1, bitLen - 1, bitLen - 1));

      sizeA = 2 + d.Random()(0);
      sizeB = 2 + d.Random()(0);
    }
    // Construct sequence of the form
    // (binary number with sizeA bits) + '+' + (binary number with sizeB bits).
    vecInput(i) = randi<arma::colvec>(
        sizeA + sizeB + 1, DistrParam(0, 1));
    // Insert special value for '+' delimiter.
    vecInput(i).at(sizeA) = 2;

    int valA = 0;
    for (size_t k = 0; k < sizeA; ++k)
    {
      valA += static_cast<int>(vecInput(i).at(k)) << k;
    }

    int valB = 0;
    for (size_t k = sizeA + 1; k < sizeA + 1 + sizeB; ++k)
    {
      valB += static_cast<int>(vecInput(i).at(k)) << (k - sizeA - 1);
    }

    int tot = valA + valB;
    std::vector<int> binarySeq;
    while (tot > 0)
    {
      binarySeq.push_back(tot & 1);
      tot >>= 1;
    }
    if (binarySeq.empty())
    {
      if (valA + valB != 0)
      {
        std::ostringstream oss;
        oss << "AddTask::Generate(): output sequence is empty "
            << "but the target sum is not 0 (=" << valA + valB << ")"
            << std::endl;
        throw std::domain_error(oss.str());
      }
      binarySeq.push_back(0);
    }
    vecLabels(i) = arma::colvec(binarySeq.size());
    for (size_t j = 0; j < binarySeq.size(); ++j)
    {
      vecLabels(i).at(j) = binarySeq[j];
    }
  }
  Binarize(vecInput, input);
  Binarize(vecLabels, labels);
  if (input.n_rows != labels.n_rows)
  {
      std::ostringstream oss;
      oss << "AddTask::Generate(): sequences after application of "
          << "Binarize() are not aligned ("
          << input.n_rows << " and " << labels.n_rows << ")"
          << std::endl;
      throw std::logic_error(oss.str());
  }
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    labels.at(i).reshape(input.at(i).n_elem, 1);
  }
}

inline void AddTask::Generate(arma::mat& input,
                              arma::mat& labels,
                              const size_t batchSize) const
{
  arma::field<arma::mat> fieldInput, fieldLabels;
  Generate(fieldInput, fieldLabels, batchSize, true);
  input.set_size(fieldInput(0).n_rows, batchSize);
  labels.set_size(fieldLabels(0).n_rows, batchSize);
  for (size_t i = 0; i < batchSize; ++i)
  {
    input.col(i) = fieldInput.at(i);
    labels.col(i) = fieldLabels.at(i);
  }
}

inline void AddTask::Binarize(const arma::field<arma::vec>& input,
                              arma::field<arma::mat>& output) const
{
  output = arma::field<arma::mat>(input.n_elem);
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output.at(i) = zeros(3, input.at(i).n_elem);
    for (size_t j = 0; j < input.at(i).n_elem; ++j)
    {
      size_t val = input.at(i).at(j);
      output.at(i).at(val, j) = 1;
    }
    output.at(i).reshape(output.at(i).n_elem, 1);
  }
}

} // namespace mlpack

#endif
