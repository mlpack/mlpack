/**
 * @file sort_impl.hpp
 * @author Konstantin Sidorov
 *
 * Implementation of SortTask class
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AUGMENTED_TASKS_SORT_IMPL_HPP
#define MLPACK_METHODS_AUGMENTED_TASKS_SORT_IMPL_HPP

#include "sort.hpp"

#include <cassert>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <ctime>

using std::vector;
using std::pair;
using std::make_pair;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

SortTask::SortTask(size_t maxLength, size_t bitLen)
  : maxLength(maxLength), bitLen(bitLen) {}

void SortTask::GenerateData(arma::field<arma::imat>& input,
                            arma::field<arma::imat>& labels,
                            size_t batchSize)
{
  input = arma::field<arma::imat>(batchSize);
  labels = arma::field<arma::imat>(batchSize);
  std::srand(unsigned(std::time(0)));
  for (size_t i = 0; i < batchSize; ++i) {
    // Random uniform length from [2..maxLength]
    size_t size = 2 + std::rand() % (maxLength - 1);
    input(i) = arma::randi<arma::imat>(size, bitLen, arma::distr_param(0, 1));
    arma::imat item_ans = arma::imat(size, bitLen);
    vector<pair<int, int>> vals(size);
    for (size_t j = 0; j < size; ++j) {
      int val = 0;
      for (size_t k = 0; k < bitLen; ++k) {
        val <<= 1;
        val += input(i).at(j, k);
      }
      vals[j] = make_pair(val, j);
    }
    sort(vals.begin(), vals.end());
    for (size_t j = 0; j < size; ++j) {
      item_ans.row(j) = input(i).row(vals[j].second);
    }
    labels(i) = item_ans;
  }

}

} // namespace tasks 
} // namespace augmented
} // namespace ann
} // namespace mlpack 
#endif
