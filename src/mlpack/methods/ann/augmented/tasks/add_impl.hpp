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

AddTask::AddTask(const size_t bitLen) : bitLen(bitLen) {
  assert(bitLen > 0);
}

void AddTask::Generate(arma::field<arma::mat>& input,
                       arma::field<arma::mat>& labels,
                       const size_t batchSize,
                       bool fixedLength = false)
{
  arma::field<arma::vec> vecInput = arma::field<arma::colvec>(batchSize);
  arma::field<arma::vec> vecLabels = arma::field<arma::colvec>(batchSize);
  size_t size_A = bitLen, size_B = bitLen;
  for (size_t i = 0; i < batchSize; ++i) {
    if (!fixedLength) {
      // Random uniform length from [2..bitLen]
      size_t size_A = RandInt(2, bitLen + 1);
      size_t size_B = RandInt(2, bitLen + 1);
    }
    // Construct sequence of the form
    // (binary number with size_A bits) + '+'
    // + (binary number with size_B bits)
    vecInput(i) = arma::randi<arma::colvec>(
      size_A + size_B + 1, arma::distr_param(0, 1));
    vecInput(i).at(size_A) = 2; // special value for '+' delimiter
    int val_A = 0;
    for (size_t k = 0; k < size_A; ++k) {
      val_A <<= 1;
      val_A += vecInput(i).at(k);
    }
    int val_B = 0;
    for (size_t k = size_A+1; k < size_A+1+size_B; ++k) {
      val_B <<= 1;
      val_B += vecInput(i).at(k);
    }
    int tot = val_A + val_B;
    vector<int> binary_seq;
    while (tot > 0) {
      binary_seq.push_back(tot & 1);
      tot >>= 1;
    }
    if (binary_seq.empty()) {
      assert(val_A + val_B == 0);
      binary_seq.push_back(0);
    }
    size_t tot_len = binary_seq.size();
    vecLabels(i) = arma::colvec(tot_len);
    for (size_t j = 0; j < tot_len; ++j) {
      vecLabels(i).at(j) = binary_seq[tot_len-j-1];
    }
  }
  input = Binarize(vecInput);
  labels = Binarize(vecLabels);
  assert(input.n_rows == labels.n_rows);
  for (size_t i = 0; i < input.n_rows; ++i) {
    labels.at(i).reshape(input.at(i).n_elem, 1);
  }
}

void AddTask::Generate(arma::mat& input, arma::mat& labels,
                       const size_t batchSize)
{
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

arma::field<arma::mat> AddTask::Binarize(arma::field<arma::vec> data) {
  arma::field<arma::mat> procData(data.n_elem);
  for (size_t i = 0; i < data.n_elem; ++i) {
    procData.at(i) = arma::zeros(3, data.at(i).n_elem);
    for (size_t j = 0; j < data.at(i).n_elem; ++j) {
      int val = data.at(i).at(j);
      procData.at(i).at(val, j) = 1;
    }
    procData.at(i).reshape(procData.at(i).n_elem, 1);
  }
  return procData;
}


} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack
#endif
