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

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using std::vector;
using std::pair;
using std::make_pair;

using namespace mlpack::ann::augmented::tasks;
using namespace mlpack::ann::augmented::scorers;

using namespace mlpack::ann;
using namespace mlpack::optimization;

// The dummy model that simply copies the sequence
// the required number of times
// (yes, no ML here, we're unit testing :)
class HardCodedCopyModel {
 public:
  HardCodedCopyModel() : nRepeats(1) {}
  void Train(
      arma::field<arma::colvec>& predictors,
      arma::field<arma::colvec>& labels) {
    auto input = predictors.at(0);
    auto output = labels.at(0);
    nRepeats = output.n_elem / input.n_elem;
  }
  void Predict(
      arma::colvec& predictors,
      arma::colvec& labels) {
    int len = predictors.n_elem;
    auto totalLen = nRepeats*len;
    labels.zeros(totalLen);
    for (size_t i = 0; i < totalLen; ++i) {
      labels.at(i) = predictors.at(i % len);
    }
  }
  void Predict(
      arma::field<arma::colvec>& predictors,
      arma::field<arma::colvec>& labels) {
    auto sz = predictors.n_elem;
    labels = arma::field<arma::colvec>(sz);
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
  void Train(arma::field<arma::mat>& predictors,
             arma::field<arma::mat>& labels)
  {
    assert(predictors.n_elem == labels.n_elem);
    bitLen = predictors.at(0).n_rows;
  }
  void Predict(
      arma::mat& predictors,
      arma::mat& labels)
  {
    size_t len = predictors.n_cols;
    labels.zeros(bitLen, len);
    vector<pair<int, int>> vals(len);
    for (size_t j = 0; j < len; ++j) {
      int val = 0;
      for (size_t k = 0; k < bitLen; ++k) {
        val <<= 1;
        val += predictors.at(k, j);
      }
      vals[j] = make_pair(val, j);
    }
    sort(vals.begin(), vals.end());
    for (size_t j = 0; j < len; ++j) {
      labels.col(j) = predictors.col(vals[j].second);
    }
  }
  void Predict(arma::field<arma::mat>& predictors,
               arma::field<arma::mat>& labels) {
    auto sz = predictors.n_elem;
    labels = arma::field<arma::mat>(sz);
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
  void Train(arma::field<arma::colvec>& predictors,
             arma::field<arma::colvec>& labels)
  {
    return;
  }
  void Predict(arma::colvec& predictors,
               arma::colvec& labels)
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
    labels = arma::colvec(tot_len);
    for (size_t j = 0; j < tot_len; ++j) {
      labels.at(j) = binary_seq[tot_len-j-1];
    }
  }
  void Predict(
      arma::field<arma::colvec>& predictors,
      arma::field<arma::colvec>& labels) {
    auto sz = predictors.n_elem;
    labels = arma::field<arma::colvec>(sz);
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
      arma::field<arma::colvec> trainPredictor, trainResponse;
      task.Generate(trainPredictor, trainResponse, 8);
      arma::field<arma::colvec> testPredictor, testResponse;
      task.Generate(testPredictor, testResponse, 8);
      HardCodedCopyModel model;
      model.Train(trainPredictor, trainResponse);
      arma::field<arma::colvec> predResponse;
      model.Predict(testPredictor, predResponse);
      // A single failure is a failure.
      if (SequencePrecision<arma::colvec>(
            testResponse, predResponse) < 0.99) {
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
    arma::field<arma::mat> trainPredictor, trainResponse;
    task.Generate(trainPredictor, trainResponse, 8);
    arma::field<arma::mat> testPredictor, testResponse;
    task.Generate(testPredictor, testResponse, 8);
    HardCodedSortModel model;
    model.Train(trainPredictor, trainResponse);
    arma::field<arma::mat> predResponse;
    model.Predict(testPredictor, predResponse);
    // A single failure is a failure.
    if (SequencePrecision<arma::mat>(testResponse, predResponse) < 0.99) {
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
    arma::field<arma::colvec> trainPredictor, trainResponse;
    task.Generate(trainPredictor, trainResponse, 8);
    arma::field<arma::colvec> testPredictor, testResponse;
    task.Generate(testPredictor, testResponse, 8);
    HardCodedAddModel model;
    model.Train(trainPredictor, trainResponse);
    arma::field<arma::colvec> predResponse;
    model.Predict(testPredictor, predResponse);
    // A single failure is a failure.
    if (SequencePrecision<arma::colvec>(testResponse, predResponse) < 0.99) {
      ok = false;
      break;
    }
  }

  BOOST_REQUIRE(ok);
}

BOOST_AUTO_TEST_CASE(LSTMBaselineTest)
{
  bool ok = true;

  const size_t outputSize = 1;
  const size_t inputSize = 1;
  const size_t rho = 2;
  const size_t maxRho = 16;

  RNN<MeanSquaredError<> > model(rho);

  model.Add<IdentityLayer<> >();
  model.Add<Linear<> >(inputSize, 20);

  model.Add<LSTM<> >(20, 7, maxRho);

  model.Add<Linear<> >(7, outputSize);
  model.Add<SigmoidLayer<> >();

  StandardSGD<decltype(model)> opt(model, 0.1, 2, -50000);

  const size_t maxLen = 3, nRepeats = 2;
  CopyTask task(maxLen, nRepeats);
  arma::field<arma::colvec> trainPredictor, trainResponse;
  const size_t trainSize = 8;
  task.Generate(trainPredictor, trainResponse, trainSize);
  const size_t testSize = 8;
  arma::field<arma::colvec> testPredictor, testResponse;
  task.Generate(testPredictor, testResponse, testSize);
  for (size_t epoch = 0; epoch < 100; ++epoch) {
    for (size_t example = 0; example < trainPredictor.n_elem; ++example) {
      arma::mat predictor(trainPredictor.at(example).n_elem, 1);
      predictor.col(0) = trainPredictor.at(example);
      arma::mat response(trainResponse.at(example).n_elem, 1);
      response.col(0) = trainResponse.at(example);
      model.Rho() = predictor.n_elem;
      model.Train(predictor, response, opt);
    }
  }

  // Evaluate the model
  std::cout << "Evaluating stage.\n";
  arma::field<arma::colvec> modelOutput(testSize);
  for (size_t example = 0; example < testSize; ++example) {
    arma::colvec softOutput;
    model.Rho() = testPredictor.at(example).n_elem;
    model.Predict(
      testPredictor.at(example),
      softOutput);
    modelOutput.at(example) = softOutput;
    for (size_t i = 0; i < softOutput.n_elem; ++i) {
      modelOutput.at(example).at(i) =
        (modelOutput.at(example).at(i)) < 0.5 ? 0 : 1;
    }
    std::cout << "Input:\n";
    std::cout << testPredictor.at(example).t() << std::endl;
    std::cout << "Model output:\n";
    std::cout << modelOutput.at(example).t() << std::endl;
    std::cout << "True output:\n";
    std::cout << testResponse.at(example).t() << std::endl;
  }
  std::cout << "Final score: "
       << SequencePrecision<arma::colvec>(testResponse, modelOutput)
       << "\n";
}

BOOST_AUTO_TEST_SUITE_END();
