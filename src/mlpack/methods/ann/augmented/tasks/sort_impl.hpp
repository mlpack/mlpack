/**
 * @file copy_impl.hpp
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

using std::vector;
using std::pair;
using std::make_pair;

namespace mlpack {
namespace ann /* Artificial Neural Network */ {
namespace augmented /* Augmented neural network */ {
namespace tasks /* Task utilities for augmented */ {

SortTask::SortTask(int maxLength, int bitLen) : maxLength(maxLength), bitLen(bitLen) {}

void SortTask::GenerateData(
  arma::field<arma::imat>& input,
  arma::field<arma::imat>& labels,
  int batchSize
) {
  input = arma::field<arma::imat>(batchSize);
  labels = arma::field<arma::imat>(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    size_t size = maxLength;
    arma::imat item = arma::randi<arma::imat>(maxLength, bitLen, arma::distr_param(0, 1));
    input(i) = item;
    arma::imat item_ans = arma::imat(maxLength, bitLen);
    vector<pair<int, int>> vals(maxLength);
    for (int j = 0; j < maxLength; ++j) {
      int val = 0;
      for (int k = 0; k < bitLen; ++k) {
        val <<= 1;
        val += item.at(j, k);
      }
      vals[j] = make_pair(val, j);
    }
    sort(vals.begin(), vals.end());
    for (int j = 0; j < maxLength; ++j) {
      item_ans.row(j) = item.row(vals[j].second);
    }
    labels(i) = item_ans;
  }
}

}
}
}
}

#endif