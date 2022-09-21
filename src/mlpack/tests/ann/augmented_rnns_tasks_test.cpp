/**
 * @file tests/augmented_rnns_tasks_test.cpp
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

#include <mlpack/core/data/binarize.hpp>

#include <mlpack/methods/ann/augmented/tasks/copy.hpp>
#include <mlpack/methods/ann/augmented/tasks/sort.hpp>
#include <mlpack/methods/ann/augmented/tasks/add.hpp>
#include <mlpack/methods/ann/augmented/tasks/score.hpp>

#include "../catch.hpp"

using std::vector;
using std::pair;
using std::make_pair;

using namespace mlpack;

using mlpack::data::Binarize;

// The dummy model that simply copies the sequence
// the required number of times
// (yes, no ML here, we're unit testing :)
class HardCodedCopyModel
{
 public:
  HardCodedCopyModel() : nRepeats(1) {}

  void Train(arma::field<arma::mat>& predictors,
             arma::field<arma::mat>& labels)
  {
    arma::mat input = predictors.at(0);
    arma::mat output = labels.at(0);
    size_t zeroCnt = 0, oneCnt = 0;
    for (size_t i = 1; i < input.n_rows; i += 2)
    {
      if (input.at(i, 0) == 0)
        ++zeroCnt;
      else
        ++oneCnt;
    }
    assert(oneCnt % zeroCnt == 0);
    nRepeats = oneCnt / zeroCnt;
  }

  void Predict(arma::mat& predictors,
               arma::mat& labels)
  {
    size_t seqLen = (predictors.n_rows / 2) / (nRepeats + 1);
    size_t outputLen = nRepeats * seqLen;
    assert(2 * (seqLen + outputLen) == predictors.n_rows);
    labels.zeros(predictors.n_rows / 2, 1);
    for (size_t i = 0; i < outputLen; ++i)
    {
      labels.at(seqLen+i) = predictors.at(2 * (i % seqLen));
    }
  }

  void Predict(arma::field<arma::mat>& predictors,
               arma::field<arma::mat>& labels)
  {
    size_t sz = predictors.n_elem;
    labels = arma::field<arma::mat>(sz);
    for (size_t i = 0; i < sz; ++i)
    {
      Predict(predictors.at(i), labels.at(i));
    }
  }

 private:
  size_t nRepeats;
};

// The dummy model that simply sorts the sequence.
class HardCodedSortModel
{
 public:
  HardCodedSortModel(size_t bitLen) : bitLen(bitLen) {}

  void Train(arma::field<arma::mat>& predictors,
             arma::field<arma::mat>& labels)
  {
    mlpack::Log::Assert(predictors.n_elem == labels.n_elem);
  }

  void Predict(arma::mat& predictors,
               arma::mat& labels)
  {
    predictors = predictors.t();
    predictors.reshape(bitLen, predictors.n_elem / bitLen);
    size_t len = predictors.n_cols;
    labels.zeros(bitLen, len);
    vector<pair<int, int>> vals(len);
    for (size_t j = 0; j < len; ++j)
    {
      int val = 0;
      for (size_t k = 0; k < bitLen; ++k)
      {
        val <<= 1;
        val += predictors.at(k, j);
      }
      vals[j] = make_pair(val, j);
    }
    sort(vals.begin(), vals.end());
    for (size_t j = 0; j < len; ++j)
    {
      labels.col(j) = predictors.col(vals[j].second);
    }
    labels.reshape(predictors.n_elem, 1);
  }

  void Predict(arma::field<arma::mat>& predictors,
               arma::field<arma::mat>& labels)
  {
    size_t sz = predictors.n_elem;
    labels = arma::field<arma::mat>(sz);
    for (size_t i = 0; i < sz; ++i)
    {
      Predict(predictors.at(i), labels.at(i));
    }
  }

 private:
  size_t bitLen;
};

// The dummy model that simply add two binary numbers.
class HardCodedAddModel
{
 public:
  HardCodedAddModel() {}

  void Train(arma::field<arma::mat>& /* predictors */,
             arma::field<arma::mat>& /* labels */)
  {
    return;
  }

  void Predict(arma::mat& predictors,
               arma::mat& labels)
  {
    assert(predictors.n_elem % 3 == 0);
    predictors = predictors.t();
    predictors.reshape(3, predictors.n_elem / 3);
    assert(predictors.n_rows == 3);
    size_t num_A = 0, num_B = 0;
    bool num = false; // True iff we have already seen the separating symbol.
    size_t cnt = 0;
    for (size_t i = 0; i < predictors.n_cols; ++i)
    {
      double digit = arma::as_scalar(arma::find(1 == predictors.col(i), 1));
      if (digit != 0 && digit != 1)
      {
        // We should not see two separators
        // since we are adding *two* numbers in the task
        assert(!num);
        num = true;
        cnt = 0;
      }
      else
      {
        if (num)
        {
          num_B += static_cast<int>(digit) << cnt;
        }
        else
        {
          num_A += static_cast<int>(digit) << cnt;
        }
        ++cnt;
      }
    }
    int total = num_A + num_B;
    vector<int> binary_seq;
    while (total > 0)
    {
      binary_seq.push_back(total & 1);
      total >>= 1;
    }
    if (binary_seq.empty())
    {
      assert(num_A + num_B == 0);
      binary_seq.push_back(0);
    }
    size_t totLen = binary_seq.size();
    labels = arma::zeros(3, totLen);
    for (size_t j = 0; j < totLen; ++j)
    {
      labels.at(binary_seq[j], j) = 1;
    }
    labels.reshape(predictors.n_elem, 1);
  }

  void Predict(
      arma::field<arma::mat>& predictors,
      arma::field<arma::mat>& labels)
  {
    size_t sz = predictors.n_elem;
    labels = arma::field<arma::mat>(sz);
    for (size_t i = 0; i < sz; ++i)
    {
      Predict(predictors.at(i), labels.at(i));
    }
  }
};


// Test of CopyTask instance generator.
// The data from generator is fed to the dummy hard-coded model above
// that should be able to solve the task perfectly.
TEST_CASE("CopyTaskTest", "[AugmentedRNNsTasks]")
{
  // Check the setup on various lengths...
  for (size_t maxLen = 2; maxLen <= 16; ++maxLen)
  {
    // .. and various numbers of repetitions.
    for (size_t nRepeats = 1; nRepeats <= 10; ++nRepeats)
    {
      CopyTask task(maxLen, nRepeats);
      arma::field<arma::mat> trainPredictor, trainResponse;
      task.Generate(trainPredictor, trainResponse, 8);
      arma::field<arma::mat> testPredictor, testResponse;
      task.Generate(testPredictor, testResponse, 8);
      HardCodedCopyModel model;
      model.Train(trainPredictor, trainResponse);
      arma::field<arma::mat> predResponse;
      model.Predict(testPredictor, predResponse);
      // A single failure is a failure.
      REQUIRE(SequencePrecision<arma::mat>(testResponse, predResponse) >= 0.99);
    }
  }
}

// Test of SortTask instance generator.
// The data from generator is fed to the dummy hard-coded model above
// that should be able to solve the task perfectly.
TEST_CASE("SortTaskTest", "[AugmentedRNNsTasks]")
{
  size_t bitLen = 5;
  for (size_t maxLen = 2; maxLen <= 16; ++maxLen)
  {
    SortTask task(maxLen, bitLen);
    arma::field<arma::mat> trainPredictor, trainResponse;
    task.Generate(trainPredictor, trainResponse, 8);
    arma::field<arma::mat> testPredictor, testResponse;
    task.Generate(testPredictor, testResponse, 8);
    HardCodedSortModel model(bitLen);
    model.Train(trainPredictor, trainResponse);
    arma::field<arma::mat> predResponse;
    model.Predict(testPredictor, predResponse);
    // A single failure is a failure.
    REQUIRE(SequencePrecision<arma::mat>(testResponse, predResponse) >= 0.99);
  }
}

// Test of AddTask instance generator.
// The data from generator is fed to the dummy hard-coded model above
// that should be able to solve the task perfectly.
TEST_CASE("AddTaskTest", "[AugmentedRNNsTasks]")
{
  for (size_t bitLen = 2; bitLen <= 16; ++bitLen)
  {
    AddTask task(bitLen);
    arma::field<arma::mat> trainPredictor, trainResponse;
    task.Generate(trainPredictor, trainResponse, 8);
    arma::field<arma::mat> testPredictor, testResponse;
    task.Generate(testPredictor, testResponse, 8);
    HardCodedAddModel model;
    model.Train(trainPredictor, trainResponse);
    arma::field<arma::mat> predResponse;
    model.Predict(testPredictor, predResponse);
    // A single failure is a failure.
    REQUIRE(SequencePrecision<arma::mat>(testResponse, predResponse) >= 0.99);
  }
}
