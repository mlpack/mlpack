/**
 * @file augmented_rnns_tasks.cpp
 * @author Konstantin Sidorov
 *
 * Tests the rtasks for augmented network models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <vector>
#include <algorithm>
#include <utility>

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/augmented/tasks/copy.hpp>
#include <mlpack/methods/ann/augmented/tasks/sort.hpp>
#include <mlpack/methods/ann/augmented/tasks/add.hpp>
#include <mlpack/methods/ann/augmented/tasks/score.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using std::vector;
using std::pair;
using std::make_pair;

using namespace mlpack::ann::augmented::tasks;
using namespace mlpack::ann::augmented::scorers;

// The dummy model that simply copies the sequence the required number of times
// (yes, no ML here, we're unit testing :)
class HardCodedCopyModel {
 public:
  HardCodedCopyModel() : nRepeats(1) {}
  void Train(
      arma::field<arma::irowvec>& predictors,
      arma::field<arma::irowvec>& labels) {
    auto input = predictors.at(0);
    auto output = labels.at(0);
    nRepeats = output.n_elem / input.n_elem;
  }
  void Predict(
      arma::irowvec& predictors,
      arma::irowvec& labels) {
    int len = predictors.n_elem;
    auto totalLen = nRepeats*len;
    labels.zeros(totalLen);
    for (size_t i = 0; i < totalLen; ++i) {
      labels.at(i) = predictors.at(i % len);
    }
  }
  void Predict(
      arma::field<arma::irowvec>& predictors,
      arma::field<arma::irowvec>& labels) {
    auto sz = predictors.n_elem;
    labels = arma::field<arma::irowvec>(sz);
    for (size_t i = 0; i < sz; ++i) {
      Predict(predictors.at(i), labels.at(i));
    }
  }

 private:
  size_t nRepeats;
};

class HardCodedSortModel {
 public:
  HardCodedSortModel() {}
  void Train(arma::field<arma::imat>& predictors,
             arma::field<arma::imat>& labels)
  {
    assert(predictors.n_elem == labels.n_elem);
    bitLen = predictors.at(0).n_cols;
  }
  void Predict(
      arma::imat& predictors,
      arma::imat& labels)
  {
    size_t len = predictors.n_rows;
    labels.zeros(len, bitLen);
    vector<pair<int, int>> vals(len);
    for (size_t j = 0; j < len; ++j) {
      int val = 0;
      for (size_t k = 0; k < bitLen; ++k) {
        val <<= 1;
        val += predictors.at(j, k);
      }
      vals[j] = make_pair(val, j);
    }
    sort(vals.begin(), vals.end());
    for (size_t j = 0; j < len; ++j) {
      labels.row(j) = predictors.row(vals[j].second);
    }
  }
  void Predict(arma::field<arma::imat>& predictors,
               arma::field<arma::imat>& labels) {
    auto sz = predictors.n_elem;
    labels = arma::field<arma::imat>(sz);
    for (size_t i = 0; i < sz; ++i) {
      Predict(predictors.at(i), labels.at(i));
    }
  }

 private:
  size_t bitLen;
};

class HardCodedAddModel {
 public:
  HardCodedAddModel() {}
  void Train(arma::field<arma::irowvec>& predictors,
             arma::field<arma::irowvec>& labels)
  {
    return;
  }
  void Predict(arma::irowvec& predictors,
               arma::irowvec& labels)
  {
    int num_A = 0, num_B = 0;
    bool num = false; // true iff we have already seen the separating symbol
    auto len = predictors.n_elem;
    for (size_t i = 0; i < len; ++i) {
      auto digit = predictors.at(i);
      if (digit != 0 && digit != 1)
      {
        // We should not see two separators
        // since we are adding *two* numbers in the task
        assert(!num);
        num = true;
      }
      else
      {
        if (num)
        {
          num_B <<= 1;
          num_B += digit;
        }
        else
        {
          num_A <<= 1;
          num_A += digit;
        }
      }
    }
    int total = num_A + num_B;
    vector<int> binary_seq;
    while (total > 0) {
      binary_seq.push_back(total & 1);
      total >>= 1;
    }
    auto tot_len = binary_seq.size();
    labels = arma::irowvec(tot_len);
    for (size_t j = 0; j < tot_len; ++j) {
      labels.at(j) = binary_seq[tot_len-j-1];
    }
  }
  void Predict(
      arma::field<arma::irowvec>& predictors,
      arma::field<arma::irowvec>& labels) {
    auto sz = predictors.n_elem;
    labels = arma::field<arma::irowvec>(sz);
    for (size_t i = 0; i < sz; ++i) {
      Predict(predictors.at(i), labels.at(i));
    }
  }

 private:
  size_t bitLen;
};

BOOST_AUTO_TEST_SUITE(AugmentedRNNsTasks);

BOOST_AUTO_TEST_CASE(CopyTaskTest)
{
  bool ok = true;
  // Check the setup on vrious lengths...
  for (size_t maxLen = 2; maxLen <= 16; ++maxLen) {
    // .. and various numbers of repetitions.
    for (size_t nRepeats = 1; nRepeats <= 10; ++nRepeats) {
      CopyTask task(maxLen, nRepeats);
      arma::field<arma::irowvec> trainPredictor, trainResponse;
      task.Generate(trainPredictor, trainResponse, 8);
      arma::field<arma::irowvec> testPredictor, testResponse;
      task.Generate(testPredictor, testResponse, 8);
      HardCodedCopyModel model;
      model.Train(trainPredictor, trainResponse);
      arma::field<arma::irowvec> predResponse;
      model.Predict(testPredictor, predResponse);
      // A single failure is a failure.
      if (SequencePrecision(testResponse, predResponse) < 0.99) {
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }
  BOOST_REQUIRE(ok);
}

BOOST_AUTO_TEST_CASE(SortTaskTest) {
  bool ok = true;
  size_t bitLen = 5;
  for (size_t maxLen = 2; maxLen <= 16; ++maxLen) {
    SortTask task(maxLen, bitLen);
    arma::field<arma::imat> trainPredictor, trainResponse;
    task.Generate(trainPredictor, trainResponse, 8);
    arma::field<arma::imat> testPredictor, testResponse;
    task.Generate(testPredictor, testResponse, 8);
    HardCodedSortModel model;
    model.Train(trainPredictor, trainResponse);
    arma::field<arma::imat> predResponse;
    model.Predict(testPredictor, predResponse);
    // A single failure is a failure.
    if (SequencePrecision(testResponse, predResponse) < 0.99) {
      ok = false;
      break;
    }
  }

  BOOST_REQUIRE(ok);
}

BOOST_AUTO_TEST_CASE(AddTaskTest) {
  bool ok = true;
  for (size_t bitLen = 2; bitLen <= 16; ++bitLen) {
    AddTask task(bitLen);
    arma::field<arma::irowvec> trainPredictor, trainResponse;
    task.Generate(trainPredictor, trainResponse, 8);
    arma::field<arma::irowvec> testPredictor, testResponse;
    task.Generate(testPredictor, testResponse, 8);
    HardCodedAddModel model;
    model.Train(trainPredictor, trainResponse);
    arma::field<arma::irowvec> predResponse;
    model.Predict(testPredictor, predResponse);
    // A single failure is a failure.
    if (SequencePrecision(testResponse, predResponse) < 0.99) {
      ok = false;
      break;
    }
  }

  BOOST_REQUIRE(ok);
}

BOOST_AUTO_TEST_SUITE_END();
