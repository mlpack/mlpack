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

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/core/data/binarize.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

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
  arma::colvec x =  arma::linspace<arma::Col<double>>(0,
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
  const size_t rho = 10;

  for (size_t trial = 0; trial < 5; ++trial)
  {
    // Generate 12 (2 * 6) noisy sines. A single sine contains rho
    // points/features.
    arma::mat input, labelsTemp;
    GenerateNoisySines(input, labelsTemp, rho, 6);

    arma::mat labels = arma::zeros<arma::mat>(rho, labelsTemp.n_cols);
    for (size_t i = 0; i < labelsTemp.n_cols; ++i)
    {
      const int value = arma::as_scalar(arma::find(
          arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
      labels.col(i).fill(value);
    }

    /**
     * Construct a network with 1 input unit, 4 hidden units and 10 output
     * units. The hidden layer is connected to itself. The network structure
     * looks like:
     *
     *  Input         Hidden        Output
     * Layer(1)      Layer(4)      Layer(10)
     * +-----+       +-----+       +-----+
     * |     |       |     |       |     |
     * |     +------>|     +------>|     |
     * |     |    ..>|     |       |     |
     * +-----+    .  +--+--+       +-----+
     *            .     .
     *            .     .
     *            .......
     */
    Add<> add(4);
    Linear<> lookup(1, 4);
    SigmoidLayer<> sigmoidLayer;
    Linear<> linear(4, 4);
    Recurrent<> recurrent(add, lookup, linear, sigmoidLayer, rho);

    RNN<> model(rho);
    model.Add<IdentityLayer<> >();
    model.Add(recurrent);
    model.Add<Linear<> >(4, 10);
    model.Add<LogSoftMax<> >();

    StandardSGD opt(0.1, 1, 500 * input.n_cols, -100);
    model.Train(input, labels, opt);

    arma::mat prediction;
    model.Predict(input, prediction);

    size_t error = 0;
    for (size_t i = 0; i < prediction.n_cols; ++i)
    {
      arma::mat singlePrediction = prediction.submat((rho - 1) * rho, i,
          rho * rho - 1, i);

      const int predictionValue = arma::as_scalar(arma::find(
          arma::max(singlePrediction.col(0)) ==
          singlePrediction.col(0), 1) + 1);

      const int targetValue = arma::as_scalar(arma::find(
          arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;

      if (predictionValue == targetValue)
      {
        error++;
      }
    }

    double classificationError = 1 - double(error) / prediction.n_cols;

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
}

/**
 * Generate a random recursive Reber grammar.
 *
 * @param transitions Recursive Reber grammar transition matrix.
 * @param averageRecursion Average recursive depth of the reber grammar.
 * @param maxRecursion Maximum recursive depth of reber grammar.
 * @param reber The generated embedded Reber grammar string.
 * @param addEnd Add ending 'E' to the generated grammar.
 */
void GenerateRecursiveReber(const arma::Mat<char>& transitions,
                            size_t averageRecursion,
                            size_t maxRecursion,
                            std::string& reber,
                            bool addEnd = true)
{
  char c = (rand() % averageRecursion) == 1 ? 'P' : 'T';

  if (maxRecursion == 1 || c == 'T')
  {
    c = 'T';
    GenerateReber(transitions, reber);
  }
  else
  {
    GenerateRecursiveReber(transitions, averageRecursion, --maxRecursion,
        reber, false);
  }

  reber = c + reber + c;

  if (addEnd)
  {
    reber = "B" + reber + "E";
  }
}

/**
 * Convert a unit vector to a Reber symbol.
 *
 * @param translation The unit vector to be converted.
 * @param symbol The converted unit vector stored as Reber symbol.
 */
template<typename MatType>
void ReberReverseTranslation(const MatType& translation, char& symbol)
{
  arma::Col<char> symbols;
  symbols << 'B' << 'T' << 'S' << 'X' << 'P' << 'V' << 'E' << arma::endr;
  const int idx = arma::as_scalar(arma::find(translation == 1, 1, "first"));

  symbol = symbols(idx);
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
 * Given a recursive Reber string, return a Reber string with all
 * reachable next symbols.
 *
 * @param transitions The Reber transistion matrix.
 * @param reber The Reber string used to generate all reachable next symbols.
 * @param nextReber All reachable next symbols.
 */
void GenerateNextRecursiveReber(const arma::Mat<char>& transitions,
                               const std::string& reber, std::string& nextReber)
{
  size_t state = 0;
  size_t numPs = 0;

  for (size_t cIndex = 0; cIndex < reber.length(); cIndex++)
  {
    char c = reber[cIndex];

    if (c == 'B' && state == 0)
    {
      state = 1;
    }
    else if (c == 'P' && state == 1)
    {
      numPs++;
      state = 1;
    }
    else if (c == 'T' && state == 1)
    {
      state = 2;
    }
    else if (c == 'B' && state == 2)
    {
      size_t pos = reber.find('E');
      if (pos != std::string::npos)
      {
        cIndex = pos;
        state = 4;
      }
      else
      {
        GenerateNextReber(transitions, reber.substr(cIndex), nextReber);
        state = 3;
      }
    }
    else if (c == 'T' && state == 4)
    {
      state = 5;
    }
    else if (c == 'P' && state == 5)
    {
      numPs--;
      state = 5;
    }
  }

  if (state == 0 || state == 2)
  {
    nextReber = "B";
  }
  else if (state == 1)
  {
    nextReber = "PT";
  }
  else if (state == 4)
  {
    nextReber = "T";
  }
  else if (state == 5)
  {
    if (numPs == 0)
    {
      nextReber = "E";
    }
    else
    {
      nextReber = "P";
    }
  }
}

/**
 * Train the specified network and the construct a Reber grammar dataset.
 */
template<typename RecurrentLayerType>
void ReberGrammarTestNetwork(size_t hiddenSize = 4,
                             bool recursive = false,
                             size_t averageRecursion = 3,
                             size_t maxRecursion = 5
                             )
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

  const size_t trainReberGrammarCount = 800;
  const size_t testReberGrammarCount = 400;

  std::string trainReber, testReber;
  arma::field<arma::mat> trainInput(1, trainReberGrammarCount);
  arma::field<arma::mat> trainLabels(1, trainReberGrammarCount);
  arma::field<arma::mat> testInput(1, testReberGrammarCount);
  arma::colvec translation;

  // Generate the training data.
  for (size_t i = 0; i < trainReberGrammarCount; i++)
  {
    if (recursive)
      GenerateRecursiveReber(transitions, 3, 5, trainReber);
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
    if (recursive)
      GenerateRecursiveReber(transitions, averageRecursion, maxRecursion,
          testReber);
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
   * +-----+    .  +--+--+       +-- ---+
   *            .     .
   *            .     .
   *            .......
   */
  // It isn't guaranteed that the recurrent network will converge in the
  // specified number of iterations using random weights. If this works 1 of 5
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;
  size_t offset = 0;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    const size_t outputSize = 7;
    const size_t inputSize = 7;

    RNN<MeanSquaredError<> > model(5);

    model.Add<Linear<> >(inputSize, hiddenSize);
    model.Add<RecurrentLayerType>(hiddenSize, hiddenSize);
    model.Add<Linear<> >(hiddenSize, outputSize);
    model.Add<SigmoidLayer<> >();

    StandardSGD opt(0.01, 50, 2, -50000);

    arma::mat inputTemp, labelsTemp;
    for (size_t i = 0; i < (10 + offset); i++)
    {
      for (size_t j = 0; j < trainReberGrammarCount; j++)
      {
        inputTemp = trainInput.at(0, j);
        labelsTemp = trainLabels.at(0, j);

        model.Rho() = inputTemp.n_elem / inputSize;
        model.Train(inputTemp, labelsTemp, opt);
      }
    }

    double error = 0;

    // Ask the network to predict the next Reber grammar in the given sequence.
    for (size_t i = 0; i < testReberGrammarCount; i++)
    {
      arma::mat prediction;
      arma::mat input = testInput.at(0, i);

      model.Rho() = input.n_elem / inputSize;
      model.Predict(input, prediction);

      const size_t reberGrammerSize = 7;
      std::string inputReber = "";

      size_t reberError = 0;

      for (size_t j = 0; j < (prediction.n_elem / reberGrammerSize); j++)
      {
        char predictedSymbol, inputSymbol;
        std::string reberChoices;

        arma::mat currentPrediction = prediction.submat(j * reberGrammerSize, 0,
            (j + 1) * reberGrammerSize - 1, 0);

        arma::umat output = (currentPrediction == (arma::ones(
            currentPrediction.n_rows, currentPrediction.n_cols) *
            arma::as_scalar(arma::max(currentPrediction))));

        ReberReverseTranslation(output, predictedSymbol);
        ReberReverseTranslation(input.submat(j * reberGrammerSize, 0, (j + 1) *
            reberGrammerSize - 1, 0), inputSymbol);
        inputReber += inputSymbol;

        if (recursive)
          GenerateNextRecursiveReber(transitions, inputReber, reberChoices);
        else
          GenerateNextReber(transitions, inputReber, reberChoices);

        if (reberChoices.find(predictedSymbol) != std::string::npos)
          reberError++;
      }

      if (reberError != (prediction.n_elem / reberGrammerSize))
        error += 1;
    }

    error /= testReberGrammarCount;
    if (error <= 0.2)
    {
      ++successes;
      break;
    }

    offset += 3;
  }

  BOOST_REQUIRE_GE(successes, 1);
}

/**
 * Train the specified networks on a Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(LSTMReberGrammarTest)
{
  ReberGrammarTestNetwork<LSTM<>>(6, false);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(LSTMRecursiveReberGrammarTest)
{
  ReberGrammarTestNetwork<LSTM<>>(22, true);
}

/**
 * Train the specified networks on a Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(GRUReberGrammarTest)
{
  ReberGrammarTestNetwork<GRU<>>(6, false);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(GRURecursiveReberGrammarTest)
{
  ReberGrammarTestNetwork<GRU<>>(20, true);
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
template<typename RecurrentLayerType>
void DistractedSequenceRecallTestNetwork()
{
  const size_t trainDistractedSequenceCount = 800;
  const size_t testDistractedSequenceCount = 400;

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
  const size_t outputSize = 3;
  const size_t inputSize = 10;
  const size_t rho = trainInput.at(0, 0).n_elem / inputSize;

  // It isn't guaranteed that the recurrent network will converge in the
  // specified number of iterations using random weights. If this works 1 of 5
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;
  size_t offset = 0;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    RNN<MeanSquaredError<> > model(rho);
    model.Add<IdentityLayer<> >();
    model.Add<Linear<> >(inputSize, 14);
    model.Add<RecurrentLayerType>(14, 7);
    model.Add<Linear<> >(7, outputSize);
    model.Add<SigmoidLayer<> >();

    StandardSGD opt(0.1, 50, 2, -50000);

    arma::mat inputTemp, labelsTemp;
    for (size_t i = 0; i < (10 + offset); i++)
    {
      for (size_t j = 0; j < trainDistractedSequenceCount; j++)
      {
        inputTemp = trainInput.at(0, j);
        labelsTemp = trainLabels.at(0, j);

        model.Train(inputTemp, labelsTemp, opt);
      }
    }

    double error = 0;

    // Ask the network to predict the targets in the given sequence at the
    // prompts.
    for (size_t i = 0; i < testDistractedSequenceCount; i++)
    {
      arma::mat output;
      arma::mat input = testInput.at(0, i);

      model.Predict(input, output);
      data::Binarize(output, output, 0.5);

      if (arma::accu(arma::abs(testLabels.at(0, i) - output)) != 0)
        error += 1;
    }

    error /= testDistractedSequenceCount;

    // Can we reproduce the results from the paper. They provide an 95% accuracy
    // on a test set of 1000 randomly selected sequences.
    // Ensure that this is within tolerance, which is at least as good as the
    // paper's results (plus a little bit for noise).
    if (error <= 0.3)
    {
      ++successes;
      break;
    }

    offset += 2;
  }

  BOOST_REQUIRE_GE(successes, 1);
}

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
BOOST_AUTO_TEST_CASE(LSTMDistractedSequenceRecallTest)
{
  DistractedSequenceRecallTestNetwork<LSTM<>>();
}

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
BOOST_AUTO_TEST_CASE(GRUDistractedSequenceRecallTest)
{
  DistractedSequenceRecallTestNetwork<GRU<>>();
}

/**
 * Make sure the RNN can be properly serialized.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  const size_t rho = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::mat input, labelsTemp;
  GenerateNoisySines(input, labelsTemp, rho, 6);

  arma::mat labels = arma::zeros<arma::mat>(rho, labelsTemp.n_cols);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
    labels.col(i).fill(value);
  }

  /**
   * Construct a network with 1 input unit, 4 hidden units and 10 output
   * units. The hidden layer is connected to itself. The network structure
   * looks like:
   *
   *  Input         Hidden        Output
   * Layer(1)      Layer(4)      Layer(10)
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   */
  Add<> add(4);
  Linear<> lookup(1, 4);
  SigmoidLayer<> sigmoidLayer;
  Linear<> linear(4, 4);
  Recurrent<> recurrent(add, lookup, linear, sigmoidLayer, rho);

  RNN<> model(rho);
  model.Add<IdentityLayer<> >();
  model.Add(recurrent);
  model.Add<Linear<> >(4, 10);
  model.Add<LogSoftMax<> >();

  StandardSGD opt(0.1, 1, input.n_cols /* 1 epoch */, -100);
  model.Train(input, labels, opt);

  // Serialize the network.
  RNN<> xmlModel(1), textModel(3), binaryModel(5);
  SerializeObjectAll(model, xmlModel, textModel, binaryModel);

  // Take predictions, check the output.
  arma::mat prediction, xmlPrediction, textPrediction, binaryPrediction;
  model.Predict(input, prediction);
  xmlModel.Predict(input, xmlPrediction);
  textModel.Predict(input, textPrediction);
  binaryModel.Predict(input, binaryPrediction);

  CheckMatrices(prediction, xmlPrediction, textPrediction, binaryPrediction);
}

BOOST_AUTO_TEST_SUITE_END();
