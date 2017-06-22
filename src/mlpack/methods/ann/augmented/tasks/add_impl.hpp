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

void AddTask::Generate(arma::field<arma::colvec>& input,
                       arma::field<arma::colvec>& labels,
                       const size_t batchSize)
{
  input = arma::field<arma::colvec>(batchSize);
  labels = arma::field<arma::colvec>(batchSize);
  for (size_t i = 0; i < batchSize; ++i) {
    // Random uniform length from [2..bitLen]
    size_t size_A = RandInt(2, bitLen + 1);
    size_t size_B = RandInt(2, bitLen + 1);
    // Construct sequence of the form
    // (binary number with size_A bits) + '+'
    // + (binary number with size_B bits)
    input(i) = arma::randi<arma::colvec>(
      size_A + size_B + 1, arma::distr_param(0, 1));
    input(i).at(size_A) = 2; // special value for '+' delimiter
    int val_A = 0;
    for (size_t k = 0; k < size_A; ++k) {
      val_A <<= 1;
      val_A += input(i).at(k);
    }
    int val_B = 0;
    for (size_t k = size_A+1; k < size_A+1+size_B; ++k) {
      val_B <<= 1;
      val_B += input(i).at(k);
    }
    int tot = val_A + val_B;
    vector<int> binary_seq;
    while (tot > 0) {
      binary_seq.push_back(tot & 1);
      tot >>= 1;
    }
    size_t tot_len = binary_seq.size();
    labels(i) = arma::colvec(tot_len);
    for (size_t j = 0; j < tot_len; ++j) {
      labels(i).at(j) = binary_seq[tot_len-j-1];
    }
  }
}

} // namespace tasks
} // namespace augmented
} // namespace ann
} // namespace mlpack
#endif
