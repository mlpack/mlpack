/**
 * @file recurrent_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the recurrent network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/recurrent_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/lstm_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>

#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
 #include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(RecurrentNetworkTest);

/**
 * Construct a 2-class dataset out of noisy sines.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */
void GenerateNoisySines(arma::mat& data,
                        arma::mat& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.3)
{
  arma::colvec x =  arma::linspace<arma::Col<double> >(0,
      points - 1, points) / points * 20.0;
  arma::colvec y1 = arma::sin(x + arma::as_scalar(arma::randu(1)) * 3.0);
  arma::colvec y2 = arma::sin(x / 2.0 + arma::as_scalar(arma::randu(1)) * 3.0);

  data = arma::zeros(points, sequences * 2);
  labels = arma::zeros(2, sequences * 2);

  for (size_t seq = 0; seq < sequences; seq++)
  {
    data.col(seq) = arma::randu(points) * noise + y1 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    labels(0, seq) = 1;

    data.col(sequences + seq) = arma::randu(points) * noise + y2 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    labels(1, sequences + seq) = 1;
  }
}

/**
 * Train the vanilla network on a larger dataset.
 */
BOOST_AUTO_TEST_CASE(SequenceClassificationTest)
{
  // It isn't guaranteed that the recurrent network will converge in the
  // specified number of iterations using random weights. If this works 1 of 5
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;

  for (size_t trial = 0; trial < 5; ++trial)
  {
    // Generate 12 (2 * 6) noisy sines. A single sine contains 10 points/features.
    arma::mat input, labels;
    GenerateNoisySines(input, labels, 10, 6);

    /*
     * Construct a network with 1 input unit, 4 hidden units and 2 output units.
     * The hidden layer is connected to itself. The network structure looks like:
     *
     *  Input         Hidden        Output
     * Layer(1)      Layer(4)      Layer(2)
     * +-----+       +-----+       +-----+
     * |     |       |     |       |     |
     * |     +------>|     +------>|     |
     * |     |    ..>|     |       |     |
     * +-----+    .  +--+--+       +-----+
     *            .     .
     *            .     .
     *            .......
     */
    LinearLayer<> linearLayer0(1, 4);
    RecurrentLayer<> recurrentLayer0(4);
    BaseLayer<LogisticFunction> inputBaseLayer;

    LinearLayer<> hiddenLayer(4, 2);
    BaseLayer<LogisticFunction> hiddenBaseLayer;

    BinaryClassificationLayer classOutputLayer;

    auto modules = std::tie(linearLayer0, recurrentLayer0, inputBaseLayer,
                            hiddenLayer, hiddenBaseLayer);

    RNN<decltype(modules), BinaryClassificationLayer, RandomInitialization,
        MeanSquaredErrorFunction> net(modules, classOutputLayer);

    SGD<decltype(net)> opt(net, 0.5, 500 * input.n_cols, -100);

    net.Train(input, labels, opt);

    arma::mat prediction;
    net.Predict(input, prediction);

    size_t error = 0;
    for (size_t i = 0; i < labels.n_cols; i++)
    {
      if (arma::sum(arma::sum(arma::abs(prediction.col(i) - labels.col(i)))) == 0)
      {
        error++;
      }
    }

    double classificationError = 1 - double(error) / labels.n_cols;
    if (classificationError <= 0.2)
    {
      ++successes;
      break;
    }
  }

  BOOST_REQUIRE_GE(successes, 1);
}

/**
 * Generate a random Reber grammar.
 *
 * For more information, see the following thesis.
 *
 * @code
 * @misc{Gers2001,
 *   author = {Felix Gers},
 *   title = {Long Short-Term Memory in Recurrent Neural Networks},
 *   year = {2001}
 * }
 * @endcode
 *
 * @param transitions Reber grammar transition matrix.
 * @param reber The generated Reber grammar string.
 */
void GenerateReber(const arma::Mat<char>& transitions, std::string& reber)
{
  size_t idx = 0;
  reber = "B";

  do
  {
    const int grammerIdx = rand() % 2;
    reber += arma::as_scalar(transitions.submat(idx, grammerIdx, idx,
        grammerIdx));

    idx = arma::as_scalar(transitions.submat(idx, grammerIdx + 2, idx,
        grammerIdx + 2)) - '0';
  } while (idx != 0);

  reber =  "BPTVVE";
}

/**
 * Generate a random embedded Reber grammar.
 *
 * @param transitions Embedded Reber grammar transition matrix.
 * @param reber The generated embedded Reber grammar string.
 */
void GenerateEmbeddedReber(const arma::Mat<char>& transitions,
                           std::string& reber)
{
  GenerateReber(transitions, reber);
  const char c = (rand() % 2) == 1 ? 'P' : 'T';
  reber = c + reber + c;
  reber = "B" + reber + "E";
}

/**
 * Convert a Reber symbol to a unit vector.
 *
 * @param symbol Reber symbol to be converted.
 * @param translation The converted symbol stored as unit vector.
 */
void ReberTranslation(const char symbol, arma::colvec& translation)
{
  arma::Col<char> symbols;
  symbols << 'B' << 'T' << 'S' << 'X' << 'P' << 'V' << 'E' << arma::endr;
  const int idx = arma::as_scalar(arma::find(symbols == symbol, 1, "first"));

  translation = arma::zeros<arma::colvec>(7);
  translation(idx) = 1;
}

/**
 * Convert a unit vector to a Reber symbol.
 *
 * @param translation The unit vector to be converted.
 * @param symbol The converted unit vector stored as Reber symbol.
 */
void ReberReverseTranslation(const arma::colvec& translation, char& symbol)
{
  arma::Col<char> symbols;
  symbols << 'B' << 'T' << 'S' << 'X' << 'P' << 'V' << 'E' << arma::endr;
  const int idx = arma::as_scalar(arma::find(translation == 1, 1, "first"));

  symbol = symbols(idx);
}

/**
 * Given a Reber string, return a Reber string with all reachable next symbols.
 *
 * @param transitions The Reber transistion matrix.
 * @param reber The Reber string used to generate all reachable next symbols.
 * @param nextReber All reachable next symbols.
 */
void GenerateNextReber(const arma::Mat<char>& transitions,
                       const std::string& reber, std::string& nextReber)
{
  size_t idx = 0;

  for (size_t grammer = 1; grammer < reber.length(); grammer++)
  {
    const int grammerIdx = arma::as_scalar(arma::find(
        transitions.row(idx) == reber[grammer], 1, "first"));

    idx = arma::as_scalar(transitions.submat(idx, grammerIdx + 2, idx,
        grammerIdx + 2)) - '0';
  }

  nextReber = arma::as_scalar(transitions.submat(idx, 0, idx, 0));
  nextReber += arma::as_scalar(transitions.submat(idx, 1, idx, 1));
}

/**
 * Given a embedded Reber string, return a embedded Reber string with all
 * reachable next symbols.
 *
 * @param transitions The Reber transistion matrix.
 * @param reber The Reber string used to generate all reachable next symbols.
 * @param nextReber All reachable next symbols.
 */
void GenerateNextEmbeddedReber(const arma::Mat<char>& transitions,
                               const std::string& reber, std::string& nextReber)
{
  if (reber.length() <= 2)
  {
    nextReber = reber.length() == 1 ? "TP" : "B";
  }
  else
  {
    size_t pos = reber.find('E');
    if (pos != std::string::npos)
    {
      nextReber = pos == reber.length() - 1 ? std::string(1, reber[1]) : "E";
    }
    else
    {
      GenerateNextReber(transitions, reber.substr(2), nextReber);
    }
  }
}

/**
 * Train the specified network and the construct a Reber grammar dataset.
 */
template<typename HiddenLayerType>
void ReberGrammarTestNetwork(HiddenLayerType& hiddenLayer0,
                             bool embedded = false)
{
  // Reber state transition matrix. (The last two columns are the indices to the
  // next path).
  arma::Mat<char> transitions;
  transitions << 'T' << 'P' << '1' << '2' << arma::endr
              << 'X' << 'S' << '3' << '1' << arma::endr
              << 'V' << 'T' << '4' << '2' << arma::endr
              << 'X' << 'S' << '2' << '5' << arma::endr
              << 'P' << 'V' << '3' << '5' << arma::endr
              << 'E' << 'E' << '0' << '0' << arma::endr;

  const size_t trainReberGrammarCount = 1000;
  const size_t testReberGrammarCount = 1000;

  std::string trainReber, testReber;
  arma::field<arma::mat> trainInput(1, trainReberGrammarCount);
  arma::field<arma::mat> trainLabels(1, trainReberGrammarCount);
  arma::field<arma::mat> testInput(1, testReberGrammarCount);
  arma::colvec translation;

  // Generate the training data.
  for (size_t i = 0; i < trainReberGrammarCount; i++)
  {
    if (embedded)
      GenerateEmbeddedReber(transitions, trainReber);
    else
      GenerateReber(transitions, trainReber);

    for (size_t j = 0; j < trainReber.length() - 1; j++)
    {
      ReberTranslation(trainReber[j], translation);
      trainInput(0, i) = arma::join_cols(trainInput(0, i), translation);

      ReberTranslation(trainReber[j + 1], translation);
      trainLabels(0, i) = arma::join_cols(trainLabels(0, i), translation);
    }
  }

  // Generate the test data.
  for (size_t i = 0; i < testReberGrammarCount; i++)
  {
    if (embedded)
      GenerateEmbeddedReber(transitions, testReber);
    else
      GenerateReber(transitions, testReber);

    for (size_t j = 0; j < testReber.length() - 1; j++)
    {
      ReberTranslation(testReber[j], translation);
      testInput(0, i) = arma::join_cols(testInput(0, i), translation);
    }
  }

  /*
   * Construct a network with 7 input units, layerSize hidden units and 7 output
   * units. The hidden layer is connected to itself. The network structure looks
   * like:
   *
   *  Input         Hidden        Output
   * Layer(7)  Layer(layerSize)   Layer(7)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   */
  const size_t lstmSize = 4 * 10;
  LinearLayer<> linearLayer0(7, lstmSize);
  RecurrentLayer<> recurrentLayer0(10, lstmSize);

  LinearLayer<>hiddenLayer(10, 7);
  BaseLayer<LogisticFunction> hiddenBaseLayer;

  BinaryClassificationLayer classOutputLayer;

  auto modules = std::tie(linearLayer0, recurrentLayer0, hiddenLayer0,
                          hiddenLayer, hiddenBaseLayer);

  RNN<decltype(modules), BinaryClassificationLayer, RandomInitialization,
      MeanSquaredErrorFunction> net(modules, classOutputLayer);

  SGD<decltype(net)> opt(net, 0.5, 2, -200);

  arma::mat inputTemp, labelsTemp;
  for (size_t i = 0; i < 15; i++)
  {
    for (size_t j = 0; j < trainReberGrammarCount; j++)
    {
      inputTemp = trainInput.at(0, j);
      labelsTemp = trainLabels.at(0, j);
      net.Train(inputTemp, labelsTemp, opt);
    }
  }

  double error = 0;

  // Ask the network to predict the next Reber grammar in the given sequence.
  for (size_t i = 0; i < testReberGrammarCount; i++)
  {
    arma::mat output;
    arma::mat input = testInput.at(0, i);

    net.Predict(input, output);

    const size_t reberGrammerSize = 7;
    std::string inputReber = "";

    size_t reberError = 0;
    for (size_t j = 0; j < (output.n_elem / reberGrammerSize); j++)
    {
      if (arma::sum(arma::sum(output.submat(j * reberGrammerSize, 0, (j + 1) *
          reberGrammerSize - 1, 0))) != 1) break;

      char predictedSymbol, inputSymbol;
      std::string reberChoices;

      ReberReverseTranslation(output.submat(j * reberGrammerSize, 0, (j + 1) *
          reberGrammerSize - 1, 0), predictedSymbol);
      ReberReverseTranslation(input.submat(j * reberGrammerSize, 0, (j + 1) *
          reberGrammerSize - 1, 0), inputSymbol);
      inputReber += inputSymbol;

      if (embedded)
        GenerateNextEmbeddedReber(transitions, inputReber, reberChoices);
      else
        GenerateNextReber(transitions, inputReber, reberChoices);

      if (reberChoices.find(predictedSymbol) != std::string::npos)
        reberError++;
    }

    if (reberError != (output.n_elem / reberGrammerSize))
      error += 1;
  }

  error /= testReberGrammarCount;
  BOOST_REQUIRE_LE(error, 0.2);
}

/**
 * Train the specified networks on a Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(ReberGrammarTest)
{
  LSTMLayer<> hiddenLayerLSTM(10);
  ReberGrammarTestNetwork(hiddenLayerLSTM);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(EmbeddedReberGrammarTest)
{
  LSTMLayer<> hiddenLayerLSTM(10);
  ReberGrammarTestNetwork(hiddenLayerLSTM, true);
}

/*
 * This sample is a simplified version of Derek D. Monner's Distracted Sequence
 * Recall task, which involves 10 symbols:
 *
 * Targets: must be recognized and remembered by the network.
 * Distractors: never need to be remembered.
 * Prompts: direct the network to give an answer.
 *
 * A single trial consists of a temporal sequence of 10 input symbols. The first
 * 8 consist of 2 randomly chosen target symbols and 6 randomly chosen
 * distractor symbols in an random order. The remaining two symbols are two
 * prompts, which direct the network to produce the first and second target in
 * the sequence, in order.
 *
 * For more information, see the following paper.
 *
 * @code
 * @misc{Monner2012,
 *   author = {Monner, Derek and Reggia, James A},
 *   title = {A generalized LSTM-like training algorithm for second-order
 *   recurrent neural networks},
 *   year = {2012}
 * }
 * @endcode
 *
 * @param input The generated input sequence.
 * @param input The generated output sequence.
 */
void GenerateDistractedSequence(arma::mat& input, arma::mat& output)
{
  input = arma::zeros<arma::mat>(10, 10);
  output = arma::zeros<arma::mat>(3, 10);

  arma::Col<size_t> index = arma::shuffle(arma::linspace<arma::Col<size_t> >(
      0, 7, 8));

  // Set the target in the input sequence and the corresponding targets in the
  // output sequence by following the correct order.
  for (size_t i = 0; i < 2; i++)
  {
    size_t idx = rand() % 2;
    input(idx, index(i)) = 1;
    output(idx, index(i) > index(i == 0) ? 9 : 8) = 1;
  }

  for (size_t i = 2; i < 8; i++)
    input(2 + rand() % 6, index(i)) = 1;


  // Set the prompts which direct the network to give an answer.
  input(8, 8) = 1;
  input(9, 9) = 1;

  input.reshape(input.n_elem, 1);
  output.reshape(output.n_elem, 1);
}

/**
 * Train the specified network and the construct distracted sequence recall
 * dataset.
 */
template<typename HiddenLayerType>
void DistractedSequenceRecallTestNetwork(HiddenLayerType& hiddenLayer0)
{
  const size_t trainDistractedSequenceCount = 1000;
  const size_t testDistractedSequenceCount = 1000;

  arma::field<arma::mat> trainInput(1, trainDistractedSequenceCount);
  arma::field<arma::mat> trainLabels(1, trainDistractedSequenceCount);
  arma::field<arma::mat> testInput(1, testDistractedSequenceCount);
  arma::field<arma::mat> testLabels(1, testDistractedSequenceCount);

  // Generate the training data.
  for (size_t i = 0; i < trainDistractedSequenceCount; i++)
    GenerateDistractedSequence(trainInput(0, i), trainLabels(0, i));

  // Generate the test data.
  for (size_t i = 0; i < testDistractedSequenceCount; i++)
    GenerateDistractedSequence(testInput(0, i), testLabels(0, i));

  /*
   * Construct a network with 10 input units, layerSize hidden units and 3
   * output units. The hidden layer is connected to itself. The network
   * structure looks like:
   *
   *  Input         Hidden        Output
   * Layer(10)  Layer(layerSize)   Layer(3)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   */
  const size_t lstmSize = 4 * 10;
  LinearLayer<> linearLayer0(10, lstmSize);
  RecurrentLayer<> recurrentLayer0(10, lstmSize);

  LinearLayer<> hiddenLayer(10, 3);
  TanHLayer<> hiddenBaseLayer;

  BinaryClassificationLayer classOutputLayer;

  auto modules = std::tie(linearLayer0, recurrentLayer0, hiddenLayer0,
                          hiddenLayer, hiddenBaseLayer);

  RNN<decltype(modules), BinaryClassificationLayer, NguyenWidrowInitialization,
      MeanSquaredErrorFunction> net(modules, classOutputLayer);

  SGD<decltype(net)> opt(net, 0.04, 2, -200);

  arma::mat inputTemp, labelsTemp;
  for (size_t i = 0; i < 40; i++)
  {
    for (size_t j = 0; j < trainDistractedSequenceCount; j++)
    {
      inputTemp = trainInput.at(0, j);
      labelsTemp = trainLabels.at(0, j);

      net.Train(inputTemp, labelsTemp, opt);
    }
  }

  double error = 0;

  // Ask the network to predict the targets in the given sequence at the
  // prompts.
  for (size_t i = 0; i < testDistractedSequenceCount; i++)
  {
    arma::mat output;
    arma::mat input = testInput.at(0, i);

    net.Predict(input, output);

    if (arma::accu(arma::abs(testLabels.at(0, i) - output)) != 0)
      error += 1;
  }

  error /= testDistractedSequenceCount;

  // Can we reproduce the results from the paper. They provide an 95% accuracy
  // on a test set of 1000 randomly selected sequences.
  // Ensure that this is within tolerance, which is at least as good as the
  // paper's results (plus a little bit for noise).
  BOOST_REQUIRE_LE(error, 0.3);
}

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
BOOST_AUTO_TEST_CASE(DistractedSequenceRecallTest)
{
  LSTMLayer<> hiddenLayerLSTMPeephole(10, true);
  DistractedSequenceRecallTestNetwork(hiddenLayerLSTMPeephole);
}

BOOST_AUTO_TEST_SUITE_END();
