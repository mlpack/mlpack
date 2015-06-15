/**
 * @file recurrent_network_test.cpp
 * @author Marcus Edel
 *
 * Tests the recurrent network.
 */
#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>

#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>

#include <mlpack/methods/ann/layer/neuron_layer.hpp>
#include <mlpack/methods/ann/layer/lstm_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/binary_classification_layer.hpp>
#include <mlpack/methods/ann/layer/multiclass_classification_layer.hpp>

#include <mlpack/methods/ann/connections/full_connection.hpp>
#include <mlpack/methods/ann/connections/self_connection.hpp>
#include <mlpack/methods/ann/connections/fullself_connection.hpp>
#include <mlpack/methods/ann/connections/connection_traits.hpp>

#include <mlpack/methods/ann/trainer/trainer.hpp>

#include <mlpack/methods/ann/ffnn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

#include <mlpack/methods/ann/performance_functions/mse_function.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::ann;


BOOST_AUTO_TEST_SUITE(RecurrentNetworkTest);

// Be careful!  When writing new tests, always get the boolean value and store
// it in a temporary, because the Boost unit test macros do weird things and
// will cause bizarre problems.

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
  NeuronLayer<LogisticFunction> inputLayer(1);
  NeuronLayer<LogisticFunction> hiddenLayer0(4);
  NeuronLayer<LogisticFunction> recurrentLayer0(hiddenLayer0.InputSize());
  NeuronLayer<LogisticFunction> hiddenLayer1(2);
  BinaryClassificationLayer outputLayer;

  RandomInitialization randInit(-0.5, 0.5);

  FullConnection<
      decltype(inputLayer),
      decltype(hiddenLayer0),
      mlpack::ann::RMSPROP,
      decltype(randInit)>
      layerCon0(inputLayer, hiddenLayer0, randInit);

  SelfConnection<
    decltype(recurrentLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(randInit)>
    layerCon2(recurrentLayer0, hiddenLayer0, randInit);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1),
      mlpack::ann::RMSPROP,
      decltype(randInit)>
      layerCon4(hiddenLayer0, hiddenLayer1, randInit);

  auto module0 = std::tie(layerCon0, layerCon2);
  auto module1 = std::tie(layerCon4);
  auto modules = std::tie(module0, module1);

  RNN<decltype(modules),
      decltype(outputLayer),
      MeanSquaredErrorFunction> net(modules, outputLayer);

  // Train the network for 1000 epochs.
  Trainer<decltype(net)> trainer(net, 1000);
  trainer.Train(input, labels, input, labels);

  // Ask the network to classify the trained input data.
  arma::colvec output;
  for (size_t i = 0; i < input.n_cols; i++)
  {
    net.Predict(input.unsafe_col(i), output);

    bool b = arma::all((output == labels.unsafe_col(i)) == 1);
    BOOST_REQUIRE_EQUAL(b, 1);
  }
}

/**
 * Train and evaluate a vanilla feed forward network and a recurrent network
 * with the specified structure and compare the two networks output and overall
 * error.
 */
template<
    typename WeightInitRule,
    typename PerformanceFunction,
    typename OutputLayerType,
    typename PerformanceFunctionType,
    typename MatType = arma::mat
>
void CompareVanillaNetworks(MatType& trainData,
                            MatType& trainLabels,
                            MatType& testData,
                            MatType& testLabels,
                            const size_t hiddenLayerSize,
                            const size_t maxEpochs,
                            WeightInitRule weightInitRule = WeightInitRule())
{
  BiasLayer<> biasLayer0(1);

  NeuronLayer<PerformanceFunction> inputLayer(trainData.n_rows);
  NeuronLayer<PerformanceFunction> hiddenLayer0(hiddenLayerSize);
  NeuronLayer<PerformanceFunction> hiddenLayer1(trainLabels.n_rows);

  OutputLayerType outputLayer;

  FullConnection<
    decltype(inputLayer),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(weightInitRule)>
    ffnLayerCon0(inputLayer, hiddenLayer0, weightInitRule);

  FullConnection<
    decltype(inputLayer),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(weightInitRule)>
    rnnLayerCon0(inputLayer, hiddenLayer0, weightInitRule);

  FullConnection<
    decltype(biasLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(weightInitRule)>
    ffnLayerCon1(biasLayer0, hiddenLayer0, weightInitRule);

  FullConnection<
    decltype(biasLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(weightInitRule)>
    rnnLayerCon1(biasLayer0, hiddenLayer0, weightInitRule);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1),
      mlpack::ann::RMSPROP,
      decltype(weightInitRule)>
      ffnLayerCon2(hiddenLayer0, hiddenLayer1, weightInitRule);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1),
      mlpack::ann::RMSPROP,
      decltype(weightInitRule)>
      rnnLayerCon2(hiddenLayer0, hiddenLayer1, weightInitRule);

  auto ffnModule0 = std::tie(ffnLayerCon0, ffnLayerCon1);
  auto ffnModule1 = std::tie(ffnLayerCon2);
  auto ffnModules = std::tie(ffnModule0, ffnModule1);

  auto rnnModule0 = std::tie(rnnLayerCon0, rnnLayerCon1);
  auto rnnModule1 = std::tie(rnnLayerCon2);
  auto rnnModules = std::tie(rnnModule0, rnnModule1);

  /*
   * Construct a feed forward network with trainData.n_rows input units,
   * hiddenLayerSize hidden units and trainLabels.n_rows output units. The
   * network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |       |     |       |     |
   * +-----+       +--+--+       +-----+
   */
  FFNN<decltype(ffnModules), decltype(outputLayer), PerformanceFunctionType>
      ffn(ffnModules, outputLayer);

  /*
   * Construct a recurrent network with trainData.n_rows input units,
   * hiddenLayerSize hidden units and trainLabels.n_rows output units. The
   * hidden layer is connected to itself. The network structure looks like:
   *
   *  Input         Hidden        Output
   *  Layer         Layer         Layer
   * +-----+       +-----+       +-----+
   * |     |       |     |       |     |
   * |     +------>|     +------>|     |
   * |     |    ..>|     |       |     |
   * +-----+    .  +--+--+       +-----+
   *            .     .
   *            .     .
   *            .......
   */
  RNN<decltype(rnnModules), decltype(outputLayer), PerformanceFunctionType>
      rnn(rnnModules, outputLayer);

  // Train the network for maxEpochs epochs or until we reach a validation error
  // of less then 0.001.
  Trainer<decltype(ffn)> ffnTrainer(ffn, maxEpochs, 1, 0.001, false);
  Trainer<decltype(rnn)> rnnTrainer(rnn, maxEpochs, 1, 0.001, false);

  for (size_t i = 0; i < 5; i++)
  {
    rnnTrainer.Train(trainData, trainLabels, testData, testLabels);
    ffnTrainer.Train(trainData, trainLabels, testData, testLabels);

    if (!arma::is_finite(ffnTrainer.ValidationError()))
      continue;

    BOOST_REQUIRE_CLOSE(ffnTrainer.ValidationError(),
        rnnTrainer.ValidationError(), 1e-3);
  }
}

/**
 * Train a vanilla feed forward and recurrent network on a sequence with len
 * one. Ideally the recurrent network should produce the same output as the
 * recurrent network. The self connection shouldn't affect the output when using
 * a sequence with a length of one.
 */
BOOST_AUTO_TEST_CASE(FeedForwardRecurrentNetworkTest)
{
  arma::mat input;
  arma::mat labels;

  RandomInitialization randInit(1, 1);

  // Test on a non-linearly separable dataset (XOR).
  input << 0 << 1 << 1 << 0 << arma::endr
        << 1 << 0 << 1 << 0 << arma::endr;
  labels << 0 << 0 << 1 << 1;

  // Vanilla neural net with logistic activation function.
  CompareVanillaNetworks<RandomInitialization,
                      LogisticFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 10, 10, randInit);

  // Vanilla neural net with identity activation function.
  CompareVanillaNetworks<RandomInitialization,
                      IdentityFunction,
                      BinaryClassificationLayer,
                      MeanSquaredErrorFunction>
      (input, labels, input, labels, 1, 1, randInit);

  // Vanilla neural net with rectifier activation function.
  CompareVanillaNetworks<RandomInitialization,
                    RectifierFunction,
                    BinaryClassificationLayer,
                    MeanSquaredErrorFunction>
    (input, labels, input, labels, 10, 10, randInit);

  // Vanilla neural net with softsign activation function.
  CompareVanillaNetworks<RandomInitialization,
                    SoftsignFunction,
                    BinaryClassificationLayer,
                    MeanSquaredErrorFunction>
    (input, labels, input, labels, 10, 10, randInit);

  // Vanilla neural net with tanh activation function.
  CompareVanillaNetworks<RandomInitialization,
                    TanhFunction,
                    BinaryClassificationLayer,
                    MeanSquaredErrorFunction>
    (input, labels, input, labels, 10, 10, randInit);
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
  NeuronLayer<LogisticFunction> inputLayer(7);
  NeuronLayer<IdentityFunction> recurrentLayer0(hiddenLayer0.OutputSize());
  NeuronLayer<LogisticFunction> hiddenLayer1(7);
  BinaryClassificationLayer outputLayer;

  NguyenWidrowInitialization randInit;

  FullConnection<
      decltype(inputLayer),
      decltype(hiddenLayer0),
      mlpack::ann::RMSPROP,
      decltype(randInit)>
      layerCon0(inputLayer, hiddenLayer0, randInit);

  FullselfConnection<
    decltype(recurrentLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(randInit)>
    layerTypeLSTM(recurrentLayer0, hiddenLayer0, randInit);

  SelfConnection<
    decltype(recurrentLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(randInit)>
    layerTypeBasis(recurrentLayer0, hiddenLayer0, randInit);

  typename std::conditional<LayerTraits<HiddenLayerType>::IsLSTMLayer,
      typename std::remove_reference<decltype(layerTypeLSTM)>::type,
      typename std::remove_reference<decltype(layerTypeBasis)>::type>::type
      layerCon2(recurrentLayer0, hiddenLayer0, randInit);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1),
      mlpack::ann::RMSPROP,
      decltype(randInit)>
      layerCon4(hiddenLayer0, hiddenLayer1, randInit);

  auto module0 = std::tie(layerCon0, layerCon2);
  auto module1 = std::tie(layerCon4);
  auto modules = std::tie(module0, module1);

  RNN<decltype(modules),
      decltype(outputLayer),
      MeanSquaredErrorFunction> net(modules, outputLayer);

  // Train the network for (500 * trainReberGrammarCount) epochs.
  Trainer<decltype(net)> trainer(net, 1, 1, 0, false);

  arma::mat inputTemp, labelsTemp;
  for (size_t i = 0; i < 100; i++)
  {
    for (size_t j = 0; j < trainReberGrammarCount; j++)
    {
      inputTemp = trainInput.at(0, j);
      labelsTemp = trainLabels.at(0, j);
      trainer.Train(inputTemp, labelsTemp, inputTemp, labelsTemp);
    }
  }

  double error = 0;

  // Ask the network to predict the next Reber grammar in the given sequence.
  for (size_t i = 0; i < testReberGrammarCount; i++)
  {
    arma::colvec output;
    arma::colvec input = testInput.at(0, i);

    net.Predict(input, output);

    const size_t reberGrammerSize = 7;
    std::string inputReber = "";

    size_t reberError = 0;
    for (size_t j = 0; j < (output.n_elem / reberGrammerSize); j++)
    {
      if (arma::sum(output.subvec(j * reberGrammerSize, (j + 1) *
          reberGrammerSize - 1)) != 1) break;

      char predictedSymbol, inputSymbol;
      std::string reberChoices;

      ReberReverseTranslation(output.subvec(j * reberGrammerSize, (j + 1) *
          reberGrammerSize - 1), predictedSymbol);
      ReberReverseTranslation(input.subvec(j * reberGrammerSize, (j + 1) *
          reberGrammerSize - 1), inputSymbol);
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

  NeuronLayer<LogisticFunction> hiddenLayerLogistic(5);
  ReberGrammarTestNetwork(hiddenLayerLogistic);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
BOOST_AUTO_TEST_CASE(EmbeddedReberGrammarTest)
{
  LSTMLayer<> hiddenLayerLSTM(10);
  ReberGrammarTestNetwork(hiddenLayerLSTM, true);

  LSTMLayer<> hiddenLayerLSTMPeephole(10, 1, true);
  ReberGrammarTestNetwork(hiddenLayerLSTMPeephole, true);
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
  NeuronLayer<LogisticFunction> inputLayer(10);
  NeuronLayer<IdentityFunction> recurrentLayer0(hiddenLayer0.OutputSize());
  NeuronLayer<LogisticFunction> hiddenLayer1(3);
  BinaryClassificationLayer outputLayer;

  NguyenWidrowInitialization randInit;

  FullConnection<
      decltype(inputLayer),
      decltype(hiddenLayer0),
      mlpack::ann::RMSPROP,
      decltype(randInit)>
      layerCon0(inputLayer, hiddenLayer0, randInit);

  FullselfConnection<
    decltype(recurrentLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(randInit)>
    layerTypeLSTM(recurrentLayer0, hiddenLayer0, randInit);

  SelfConnection<
    decltype(recurrentLayer0),
    decltype(hiddenLayer0),
    mlpack::ann::RMSPROP,
    decltype(randInit)>
    layerTypeBasis(recurrentLayer0, hiddenLayer0, randInit);

  typename std::conditional<LayerTraits<HiddenLayerType>::IsLSTMLayer,
      typename std::remove_reference<decltype(layerTypeLSTM)>::type,
      typename std::remove_reference<decltype(layerTypeBasis)>::type>::type
      layerCon2(recurrentLayer0, hiddenLayer0, randInit);

  FullConnection<
      decltype(hiddenLayer0),
      decltype(hiddenLayer1),
      mlpack::ann::RMSPROP,
      decltype(randInit)>
      layerCon4(hiddenLayer0, hiddenLayer1, randInit);

  auto module0 = std::tie(layerCon0, layerCon2);
  auto module1 = std::tie(layerCon4);
  auto modules = std::tie(module0, module1);

  RNN<decltype(modules),
      decltype(outputLayer),
      MeanSquaredErrorFunction> net(modules, outputLayer);

  // Train the network for (500 * trainDistractedSequenceCount) epochs.
  Trainer<decltype(net)> trainer(net, 1, 1, 0, false);

  arma::mat inputTemp, labelsTemp;
  for (size_t i = 0; i < 100; i++)
  {
    for (size_t j = 0; j < trainDistractedSequenceCount; j++)
    {
      inputTemp = trainInput.at(0, j);
      labelsTemp = trainLabels.at(0, j);

      trainer.Train(inputTemp, labelsTemp, inputTemp, labelsTemp);
    }
  }

  double error = 0;

  // Ask the network to predict the targets in the given sequence at the
  // prompts.
  for (size_t i = 0; i < testDistractedSequenceCount; i++)
  {
    arma::colvec output;
    arma::colvec input = testInput.at(0, i);

    net.Predict(input, output);

    if (arma::sum(arma::abs(testLabels.at(0, i) - output)) != 0)
      error += 1;
  }

  error /= testDistractedSequenceCount;

  // Can we reproduce the results from the paper. They provide an 95% accuracy
  // on a test set of 1000 randomly selected sequences.
  // Ensure that this is within tolerance, which is at least as good as the
  // paper's results (plus a little bit for noise).
  BOOST_REQUIRE_LE(error, 0.1);
}

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
BOOST_AUTO_TEST_CASE(DistractedSequenceRecallTest)
{
  LSTMLayer<> hiddenLayerLSTM(10, 10);
  DistractedSequenceRecallTestNetwork(hiddenLayerLSTM);

  LSTMLayer<> hiddenLayerLSTMPeephole(10, 1, true);
  DistractedSequenceRecallTestNetwork(hiddenLayerLSTM);
}

BOOST_AUTO_TEST_SUITE_END();