/**
 * @file tests/recurrent_network_test.cpp
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

#include <mlpack/methods/ann/ann.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "custom_layer.hpp"

using namespace mlpack;
using namespace ens;

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
  symbols = { 'B', 'T', 'S', 'X', 'P', 'V', 'E' };
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
  symbols = { 'B', 'T', 'S', 'X', 'P', 'V', 'E' };
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
                                const std::string& reber,
                                std::string& nextReber)
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
 * @brief Creates the reber grammar data for tests.
 *
 * @param trainInput The train data
 * @param trainLabels The train labels
 * @param testInput The test input
 * @param recursive whether recursive Reber
 * @param trainReberGrammarCount The number of training set
 * @param testReberGrammarCount The number of test set
 * @param averageRecursion Average recursion
 * @param maxRecursion Max recursion
 * @return arma::Mat<char> The Reber state translation to be used.
 */
arma::Mat<char> GenerateReberGrammarData(
                              arma::field<arma::mat>& trainInput,
                              arma::field<arma::mat>& trainLabels,
                              arma::field<arma::mat>& testInput,
                              bool recursive = false,
                              const size_t trainReberGrammarCount = 700,
                              const size_t testReberGrammarCount = 250,
                              const size_t averageRecursion = 3,
                              const size_t maxRecursion = 5)
{
  // Reber state transition matrix. (The last two columns are the indices to the
  // next path).
  arma::Mat<char> transitions;
  transitions = { { 'T', 'P', '1', '2' },
                  { 'X', 'S', '3', '1' },
                  { 'V', 'T', '4', '2' },
                  { 'X', 'S', '2', '5' },
                  { 'P', 'V', '3', '5' },
                  { 'E', 'E', '0', '0' } };


  std::string trainReber, testReber;

  arma::colvec translation;

  // Generate the training data.
  for (size_t i = 0; i < trainReberGrammarCount; ++i)
  {
    if (recursive)
      GenerateRecursiveReber(transitions, 3, 5, trainReber);
    else
      GenerateReber(transitions, trainReber);

    for (size_t j = 0; j < trainReber.length() - 1; ++j)
    {
      ReberTranslation(trainReber[j], translation);
      trainInput(0, i) = arma::join_cols(trainInput(0, i), translation);

      ReberTranslation(trainReber[j + 1], translation);
      trainLabels(0, i) = arma::join_cols(trainLabels(0, i), translation);
    }
  }

  // Generate the test data.
  for (size_t i = 0; i < testReberGrammarCount; ++i)
  {
    if (recursive)
      GenerateRecursiveReber(transitions, averageRecursion, maxRecursion,
          testReber);
    else
      GenerateReber(transitions, testReber);

    for (size_t j = 0; j < testReber.length() - 1; ++j)
    {
      ReberTranslation(testReber[j], translation);
      testInput(0, i) = arma::join_cols(testInput(0, i), translation);
    }
  }

  return transitions;
}

/**
 * Train the specified network and the construct a Reber grammar dataset.
 */
template<typename ModelType>
void ReberGrammarTestNetwork(ModelType& model,
                             const bool recursive = false,
                             const size_t averageRecursion = 3,
                             const size_t maxRecursion = 5,
                             const size_t iterations = 10,
                             const size_t trials = 5)
{
  const size_t trainReberGrammarCount = 700;
  const size_t testReberGrammarCount = 250;

  arma::field<arma::mat> trainInput(1, trainReberGrammarCount);
  arma::field<arma::mat> trainLabels(1, trainReberGrammarCount);
  arma::field<arma::mat> testInput(1, testReberGrammarCount);

  arma::Mat<char> transitions =
                  GenerateReberGrammarData(trainInput,
                                           trainLabels,
                                           testInput,
                                           recursive,
                                           trainReberGrammarCount,
                                           testReberGrammarCount,
                                           averageRecursion,
                                           maxRecursion);

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
  const size_t inputSize = 7;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    // Reset model before using for next trial.
    model.Reset();
    MomentumSGD opt(0.06, 50, 2, -50000);

    arma::cube inputTemp, labelsTemp;
    for (size_t iteration = 0; iteration < (iterations + offset); iteration++)
    {
      for (size_t j = 0; j < trainReberGrammarCount; ++j)
      {
        // Each sequence may be a different length, so we need to extract them
        // manually.  We will reshape them into a cube with each slice equal to
        // a time step.
        inputTemp = arma::cube(trainInput.at(0, j).memptr(), inputSize, 1,
            trainInput.at(0, j).n_elem / inputSize, false, true);
        labelsTemp = arma::cube(trainLabels.at(0, j).memptr(), inputSize, 1,
            trainInput.at(0, j).n_elem / inputSize, false, true);

        model.Rho() = inputTemp.n_elem / inputSize;
        model.Train(inputTemp, labelsTemp, opt);
        opt.ResetPolicy() = false;
      }
    }

    double error = 0;

    // Ask the network to predict the next Reber grammar in the given sequence.
    for (size_t i = 0; i < testReberGrammarCount; ++i)
    {
      arma::cube prediction;
      arma::cube input(testInput.at(0, i).memptr(), inputSize, 1,
          testInput.at(0, i).n_elem / inputSize, false, true);

      model.Rho() = input.n_elem / inputSize;
      model.Predict(input, prediction);

      const size_t reberGrammerSize = 7;
      std::string inputReber = "";

      size_t reberError = 0;

      for (size_t j = 0; j < (prediction.n_elem / reberGrammerSize); ++j)
      {
        char predictedSymbol, inputSymbol;
        std::string reberChoices;

        arma::umat output = (prediction.slice(j) == (arma::ones(
            reberGrammerSize, 1) *
            arma::as_scalar(arma::max(prediction.slice(j)))));

        ReberReverseTranslation(output, predictedSymbol);
        ReberReverseTranslation(input.slice(j), inputSymbol);
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
    if (error <= 0.3)
    {
      ++successes;
      break;
    }

    offset += 3;
  }

  REQUIRE(successes >= 1);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
TEST_CASE("LSTMReberGrammarTest", "[RecurrentNetworkTest]")
{
  RNN<MeanSquaredError<> > model(5);
  model.Add<Linear<> >(7, 10);
  model.Add<LSTM<> >(10, 10);
  model.Add<Linear<> >(10, 7);
  model.Add<SigmoidLayer<> >();
  ReberGrammarTestNetwork(model, false);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
TEST_CASE("FastLSTMReberGrammarTest", "[RecurrentNetworkTest]")
{
  RNN<MeanSquaredError<> > model(5);
  model.Add<Linear<> >(7, 8);
  model.Add<FastLSTM<> >(8, 8);
  model.Add<Linear<> >(8, 7);
  model.Add<SigmoidLayer<> >();
  ReberGrammarTestNetwork(model, false);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
TEST_CASE("GRURecursiveReberGrammarTest", "[RecurrentNetworkTest]")
{
  RNN<MeanSquaredError<> > model(5);
  model.Add<Linear<> >(7, 16);
  model.Add<GRU<> >(16, 16);
  model.Add<Linear<> >(16, 7);
  model.Add<SigmoidLayer<> >();
  ReberGrammarTestNetwork(model, true, 3, 5, 10, 7);
}

/**
 * Train BLSTM on an embedded Reber grammar dataset.
 */
TEST_CASE("BRNNReberGrammarTest", "[RecurrentNetworkTest]")
{
  BRNN<MeanSquaredError<>, AddMerge<>, SigmoidLayer<> > model(5);
  model.Add<Linear<> >(7, 10);
  model.Add<LSTM<> >(10, 10);
  model.Add<Linear<> >(10, 7);
  ReberGrammarTestNetwork(model, false, 3, 5, 1);
}

/**
 * Test RNN with a custom layer.
 */
void ReberGrammarTestCustomNetwork(const size_t hiddenSize = 4,
                                   const bool recursive = false,
                                   const size_t iterations = 10)
{
  const size_t trainReberGrammarCount = 700;
  const size_t testReberGrammarCount = 250;

  arma::field<arma::mat> trainInput(1, trainReberGrammarCount);
  arma::field<arma::mat> trainLabels(1, trainReberGrammarCount);
  arma::field<arma::mat> testInput(1, testReberGrammarCount);

  arma::Mat<char> transitions =
                  GenerateReberGrammarData(trainInput,
                                           trainLabels,
                                           testInput,
                                           recursive,
                                           trainReberGrammarCount,
                                           testReberGrammarCount);

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
  // specified number of iterations using random weights. If this works 1 of 10
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;
  size_t offset = 0;
  for (size_t trial = 0; trial < 10; ++trial)
  {
    const size_t outputSize = 7;
    const size_t inputSize = 7;

    RNN<MeanSquaredError<>, RandomInitialization, CustomLayer<> > model(5);
    model.Add<Linear<> >(inputSize, hiddenSize);
    model.Add<GRU<> >(hiddenSize, hiddenSize);
    model.Add<Linear<> >(hiddenSize, outputSize);
    model.Add<CustomLayer<> >();
    MomentumSGD opt(0.06, 50, 2, -50000);

    arma::cube inputTemp, labelsTemp;
    for (size_t iteration = 0; iteration < (iterations + offset); iteration++)
    {
      for (size_t j = 0; j < trainReberGrammarCount; ++j)
      {
        // Each sequence may be a different length, so we need to extract them
        // manually.  We will reshape them into a cube with each slice equal to
        // a time step.
        inputTemp = arma::cube(trainInput.at(0, j).memptr(), inputSize, 1,
            trainInput.at(0, j).n_elem / inputSize, false, true);
        labelsTemp = arma::cube(trainLabels.at(0, j).memptr(), inputSize, 1,
            trainInput.at(0, j).n_elem / inputSize, false, true);

        model.Rho() = inputTemp.n_elem / inputSize;
        model.Train(inputTemp, labelsTemp, opt);
        opt.ResetPolicy() = false;
      }
    }

    double error = 0;

    // Ask the network to predict the next Reber grammar in the given sequence.
    for (size_t i = 0; i < testReberGrammarCount; ++i)
    {
      arma::cube prediction;
      arma::cube input(testInput.at(0, i).memptr(), inputSize, 1,
          testInput.at(0, i).n_elem / inputSize, false, true);

      model.Rho() = input.n_elem / inputSize;
      model.Predict(input, prediction);

      const size_t reberGrammerSize = 7;
      std::string inputReber = "";

      size_t reberError = 0;

      for (size_t j = 0; j < (prediction.n_elem / reberGrammerSize); ++j)
      {
        char predictedSymbol, inputSymbol;
        std::string reberChoices;

        arma::umat output = (prediction.slice(j) == (arma::ones(
            reberGrammerSize, 1) *
            arma::as_scalar(arma::max(prediction.slice(j)))));

        ReberReverseTranslation(output, predictedSymbol);
        ReberReverseTranslation(input.slice(j), inputSymbol);
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
    if (error <= 0.35)
    {
      ++successes;
      break;
    }

    offset += 3;
  }

  REQUIRE(successes >= 1);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
TEST_CASE("CustomRecursiveReberGrammarTest", "[RecurrentNetworkTest]")
{
  ReberGrammarTestCustomNetwork(16, true);
}
