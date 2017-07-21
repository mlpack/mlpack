/**
 * @file add_impl.hpp
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
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

AddTask::AddTask(const size_t bitLen) : bitLen(bitLen)
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

const void AddTask::Generate(arma::field<arma::mat>& input,
                             arma::field<arma::mat>& labels,
                             const size_t batchSize,
                             bool fixedLength)
{
  input = arma::field<arma::mat>(batchSize);
  labels = arma::field<arma::mat>(batchSize);
  size_t sizeA = bitLen, sizeB = bitLen;
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (!fixedLength)
    {
      // Generate random uniform length from [2..maxLength].
      sizeA = mlpack::math::RandInt(2, bitLen + 1);
      sizeB = mlpack::math::RandInt(2, bitLen + 1);
    }
    // Construct sequence of the form
    // (binary number with sizeA bits) + '+'
    // + (binary number with sizeB bits).
    input(i) = arma::randi<arma::mat>(sizeA + sizeB + 1,
                                      1,
                                      arma::distr_param(0, 1));
    // Adding leading 1 to make the distribution over numbers uniform.
    input(i).at(sizeA - 1, 0) = 1;
    input(i).at(sizeA + sizeB, 0) = 1;
    // Insert special value for '+' delimiter.
    labels(i) = arma::zeros(sizeA + sizeB + 1, 1);
    input(i).at(sizeA, 0) = 0.5;
    
    int valA = 0;
    for (size_t k = 0; k < sizeA; ++k)
    {
      valA += static_cast<int>(input(i).at(k, 0)) << k;
    }

    int valB = 0;
    for (size_t k = sizeA + 1; k < sizeA + 1 + sizeB; ++k)
    {
      valB += static_cast<int>(input(i).at(k, 0)) << (k - sizeA - 1);
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
      if (valA + valB != 0) {
        std::ostringstream oss;
        oss << "AddTask::Generate(): output sequence is empty "
            << "but the target sum is not 0 (=" << valA + valB << ")"
            << std::endl;
        throw std::domain_error(oss.str());
      }
      binarySeq.push_back(0);
    }
    for (size_t j = 0; j < binarySeq.size(); ++j)
    {
      labels(i).at(j, 0) = binarySeq[j];
    }
  }
}

const void AddTask::Generate(arma::mat& input,
                             arma::mat& labels,
                             const size_t batchSize)
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

const void AddTask::Binarize(const arma::field<arma::vec>& input,
                             arma::field<arma::mat>& output)
{
  output = arma::field<arma::mat>(input.n_elem);
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    output.at(i) = arma::conv_to<arma::mat>::from(input.at(i));
  }
}


} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack
#endif
