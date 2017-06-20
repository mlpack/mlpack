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

#include <fstream>

#include <mlpack/core.hpp>

#include <mlpack/core/data/binarize.hpp>

#include <mlpack/methods/ann/augmented/tasks/copy.hpp>
#include <mlpack/methods/ann/augmented/tasks/sort.hpp>
#include <mlpack/methods/ann/augmented/tasks/add.hpp>
#include <mlpack/methods/ann/augmented/tasks/score.hpp>

#include <mlpack/core/optimizers/adam/adam.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/leaky_relu.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using std::vector;
using std::pair;
using std::make_pair;

using std::ofstream;

using namespace mlpack::ann::augmented::tasks;
using namespace mlpack::ann::augmented::scorers;

using namespace mlpack::ann;
using namespace mlpack::optimization;

using mlpack::data::Binarize;

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


BOOST_AUTO_TEST_CASE(LSTMBaselineTestCopyRepeatRepr)
{
  ofstream fout;
  fout.open("output-aug.log");
  fout << "Report for augmented representation.\n";
  fout << "Training epochs = " << 20 << "\n";
  
  int maxNRepeat[] = {10, 10, 10, 6, 4, 3, 3, 3, 2, 2, 2};
  for (size_t maxLen = 2; maxLen <= 8; ++maxLen) {
    for (size_t nRepeats = 1; nRepeats <= maxNRepeat[maxLen]; ++nRepeats) {
      bool ok = true;

      const size_t outputSize = 1;
      const size_t inputSize = 2;
      const size_t rho = 2;
      const size_t maxRho = 128;

      RNN<MeanSquaredError<> > model(rho);

      model.Add<IdentityLayer<> >();
      model.Add<Linear<> >(inputSize, 30);
      model.Add<LSTM<> >(30, 15, maxRho);
      model.Add<LeakyReLU<> >();
      model.Add<Linear<> >(15, outputSize);
      model.Add<SigmoidLayer<> >();

      Adam<decltype(model)> opt(model);

      CopyTask task(maxLen, nRepeats);

      arma::field<arma::colvec> trainPredictor, trainResponse;
      size_t trainSize = 15 + 5 * maxLen;
      task.Generate(trainPredictor, trainResponse, trainSize);
      size_t testSize = 15 + 5 * maxLen;
      arma::field<arma::colvec> testPredictor, testResponse;
      task.Generate(testPredictor, testResponse, testSize);
      for (size_t epoch = 0; epoch < 20; ++epoch) {
        for (size_t example = 0; example < trainPredictor.n_elem; ++example) {
          size_t totSize =
            trainPredictor.at(example).n_elem + trainResponse.at(example).n_elem;
          arma::mat predictor = arma::zeros(totSize, 2);
          predictor.col(0).rows(0,trainPredictor.at(example).n_elem-1) =
            trainPredictor.at(example);
          predictor.col(1).rows(trainPredictor.at(example).n_elem,totSize-1) =
            arma::ones(totSize-trainPredictor.at(example).n_elem);
          predictor = predictor.t();
          predictor.reshape(predictor.n_elem, 1);
          arma::mat response = arma::zeros(totSize, 1);
          response.col(0).rows(trainPredictor.at(example).n_elem,totSize-1) =
            trainResponse.at(example);
          model.Rho() = totSize;
          model.Train(predictor, response, opt);
        }
        std::cerr << "Finished running training epoch #"
                  << epoch+1 << "\n";
      }

      arma::field<arma::colvec> modelOutput(testSize);
      for (size_t example = 0; example < testSize; ++example) {
        arma::colvec softOutput;
        size_t totSize =
            testPredictor.at(example).n_elem + testResponse.at(example).n_elem;
        arma::mat predictor = arma::zeros(totSize, 2);
        predictor.col(0).rows(0,testPredictor.at(example).n_elem-1) =
          testPredictor.at(example);
        assert(predictor.n_rows == totSize);
        predictor.col(1).rows(testPredictor.at(example).n_elem, totSize-1) =
          arma::ones(totSize-testPredictor.at(example).n_elem);
        predictor = predictor.t();
        predictor.reshape(predictor.n_elem, 1);
        model.Rho() = totSize;
        model.Predict(
          predictor,
          softOutput);
        modelOutput.at(example) = softOutput.rows(
          testPredictor.at(example).n_elem,
          softOutput.n_rows-1);
        Binarize<double>(modelOutput.at(example), modelOutput.at(example), 0.5);
        std::cerr  << "Predictor:\n"
                   << predictor
                   << "Model response:\n"
                   << softOutput;
        std::cerr  << "Original data:\n"
                   << testPredictor.at(example)
                   << "Test response:\n"
                   << testResponse.at(example);
      }
      std::cerr << "Final score for ("
                << maxLen << ","
                << nRepeats << "): "
                << SequencePrecision<arma::colvec>(testResponse, modelOutput)
                << "\n";

      fout      << "Final score for ("
                << maxLen << ","
                << nRepeats << "): "
                << SequencePrecision<arma::colvec>(testResponse, modelOutput)
                << "\n";
      fout << "Sample size = " << trainSize << "\n";
      fout.flush();
    }
  }
  fout.close();
}

arma::field<arma::colvec> binarizeAdd(arma::field<arma::colvec> data) {
  arma::field<arma::colvec> procData(data.n_elem);
  for (size_t i = 0; i < data.n_elem; ++i) {
    arma::colvec temp = arma::zeros(
      3, data.at(i).n_elem);
    for (size_t j = 0; j < data.at(i).n_elem; ++j) {
      int val = data.at(i).at(j);
      temp.at(val, j) = 1;
    }
    temp.reshape(temp.n_elem, 1);
    procData.at(i) = temp;
  }
  return procData;
}

BOOST_AUTO_TEST_SUITE_END();
