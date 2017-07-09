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

#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <ctime>

using std::vector;
using std::pair;
using std::make_pair;

using mlpack::math::RandInt;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

AddTask::AddTask(const size_t bitLen) : bitLen(bitLen)
{
  assert(bitLen > 0);
}

void AddTask::Generate(arma::field<arma::mat>& input,
                       arma::field<arma::mat>& labels,
                       const size_t batchSize,
                       bool fixedLength)
{
  arma::field<arma::vec> vecInput = arma::field<arma::colvec>(batchSize);
  arma::field<arma::vec> vecLabels = arma::field<arma::colvec>(batchSize);
  size_t size_A = bitLen, size_B = bitLen;
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (!fixedLength)
    {
      // Generate random uniform length from [2..maxLength].
      size_t size_A = RandInt(2, bitLen + 1);
      size_t size_B = RandInt(2, bitLen + 1);
    }
    // Construct sequence of the form
    // (binary number with size_A bits) + '+'
    // + (binary number with size_B bits).
    vecInput(i) = arma::randi<arma::colvec>(
        size_A + size_B + 1, arma::distr_param(0, 1));
    // Insert special value for '+' delimiter.
    vecInput(i).at(size_A) = 2;
    int val_A = 0;
    for (size_t k = 0; k < size_A; ++k)
    {
      val_A <<= 1;
      val_A += vecInput(i).at(k);
    }
    int val_B = 0;
    for (size_t k = size_A+1; k < size_A+1+size_B; ++k)
    {
      val_B <<= 1;
      val_B += vecInput(i).at(k);
    }
    int tot = val_A + val_B;
    vector<int> binary_seq;
    while (tot > 0)
    {
      binary_seq.push_back(tot & 1);
      tot >>= 1;
    }
    if (binary_seq.empty())
    {
      assert(val_A + val_B == 0);
      binary_seq.push_back(0);
    }
    size_t tot_len = binary_seq.size();
    vecLabels(i) = arma::colvec(tot_len);
    for (size_t j = 0; j < tot_len; ++j)
    {
      vecLabels(i).at(j) = binary_seq[tot_len-j-1];
    }
  }
  Binarize(vecInput, input);
  Binarize(vecLabels, labels);
  assert(input.n_rows == labels.n_rows);
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    labels.at(i).reshape(input.at(i).n_elem, 1);
  }
}

void AddTask::Generate(arma::mat& input, arma::mat& labels,
                       const size_t batchSize)
{
  arma::field<arma::mat> fieldInput, fieldLabels;
  Generate(fieldInput, fieldLabels, batchSize, true);
  size_t cols = batchSize;
  input.set_size(fieldInput(0).n_rows, cols);
  labels.set_size(fieldLabels(0).n_rows, cols);
  for (size_t i = 0; i < cols; ++i)
  {
    input.col(i) = fieldInput.at(i);
    labels.col(i) = fieldLabels.at(i);
  }
}

void AddTask::Binarize(const arma::field<arma::vec>& input,
                       arma::field<arma::mat>& output)
{
  arma::field<arma::mat> procData(input.n_elem);
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    procData.at(i) = arma::zeros(3, input.at(i).n_elem);
    for (size_t j = 0; j < input.at(i).n_elem; ++j)
    {
      int val = input.at(i).at(j);
      procData.at(i).at(val, j) = 1;
    }
    procData.at(i).reshape(procData.at(i).n_elem, 1);
  }
  output = procData;
}


} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack
#endif
