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

#include "../catch.hpp"
#include "../serialization.hpp"

using namespace mlpack;
using namespace ens;

/**
 * Construct a 2-class dataset out of noisy sines.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */
void GenerateNoisySines(arma::cube& data,
                        arma::mat& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.3)
{
  arma::colvec x =  arma::linspace<arma::colvec>(0, points - 1, points) /
      points * 20.0;
  arma::colvec y1 = arma::sin(x + arma::as_scalar(arma::randu(1)) * 3.0);
  arma::colvec y2 = arma::sin(x / 2.0 + arma::as_scalar(arma::randu(1)) * 3.0);

  data = arma::zeros(1 /* single dimension */, sequences * 2, points);
  labels = arma::zeros(2 /* 2 classes */, sequences * 2);

  for (size_t seq = 0; seq < sequences; seq++)
  {
    arma::vec sequence = arma::randu(points) * noise + y1 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, seq, i) = sequence[i];

    labels(0, seq) = 1;

    sequence = arma::randu(points) * noise + y2 +
        arma::as_scalar(arma::randu(1) - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, sequences + seq, i) = sequence[i];

    labels(1, sequences + seq) = 1;
  }
}

/**
 * Construct dataset for sine wave prediction.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */
void GenerateSines(arma::cube& data,
                   arma::cube& labels,
                   const size_t sequences,
                   const size_t len)
{
  arma::vec x = arma::sin(arma::linspace<arma::colvec>(0,
      sequences + len, sequences + len));
  data.set_size(1, len, sequences);
  labels.set_size(1, 1, sequences);

  for (size_t i = 0; i < sequences; ++i)
  {
    data.slice(i) = arma::reshape(x.subvec(i, i + len), 1, len);
    labels.slice(i) = x(i + len);
  }
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

  arma::uvec index = arma::shuffle(arma::linspace<arma::uvec>(0, 7, 8));

  // Set the target in the input sequence and the corresponding targets in the
  // output sequence by following the correct order.
  for (size_t i = 0; i < 2; ++i)
  {
    size_t idx = rand() % 2;
    input(idx, index(i)) = 1;
    output(idx, index(i) > index(i == 0) ? 9 : 8) = 1;
  }

  for (size_t i = 2; i < 8; ++i)
    input(2 + rand() % 6, index(i)) = 1;

  // Set the prompts which direct the network to give an answer.
  input(8, 8) = 1;
  input(9, 9) = 1;

  input.reshape(input.n_elem, 1);
  output.reshape(output.n_elem, 1);
}

/**
 * Create a simple recurrent neural network for the noisy sines task, and
 * require that it produces the exact same network for a few batch sizes.
 */
template<typename RecurrentLayerType>
void BatchSizeTest()
{
  const size_t T = 50;
  const size_t bpttTruncate = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input;
  arma::mat labelsTemp;
  GenerateNoisySines(input, labelsTemp, 4, 5);

  arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, T);
  for (size_t i = 0; i < labelsTemp.n_cols; ++i)
  {
    const int value = arma::as_scalar(arma::find(
        arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
    labels.tube(0, i).fill(value);
  }

  RNN<> model(bpttTruncate);
  model.Add<Linear>(100);
  model.Add<Sigmoid>();
  model.Add<RecurrentLayerType>(10);
  model.Add<Sigmoid>();
  model.Add<Linear>(10);
  model.Add<Sigmoid>();

  model.Reset(1);
  arma::mat initParams = model.Parameters();

  StandardSGD opt(1e-5, 1, 5, -100, false);
  model.Train(input, labels, opt);

  // This is trained with one point.
  arma::mat outputParams = model.Parameters();

  model.Reset(1);
  model.Parameters() = initParams;
  opt.BatchSize() = 2;
  model.Train(input, labels, opt);

  CheckMatrices(outputParams, model.Parameters(), 1);

  model.Reset(1);
  model.Parameters() = initParams;
  opt.BatchSize() = 5;
  model.Train(input, labels, opt);

  CheckMatrices(outputParams, model.Parameters(), 1);
}

/**
 * Ensure LSTMs work with larger batch sizes.
 */
TEST_CASE("LSTMBatchSizeTest", "[RecurrentNetworkTest]")
{
  BatchSizeTest<Linear>();
}

/**
 * @brief Generates noisy sine wave and outputs the data and the labels that
 *        can be used directly for training and testing with RNN.
 *
 * @param data The data points as output
 * @param labels The expected values as output
 * @param rho The size of the sequence of each data point
 * @param outputSteps How many output steps to consider for every rho inputs
 * @param dataPoints  The number of generated data points. The actual generated
 *        data points may be more than this to adjust to the outputSteps. But at
 *        the minimum these many data points will be generated.
 * @param gain The gain on the amplitude
 * @param freq The frquency of the sine wave
 * @param phase The phase shift if any
 * @param noisePercent The percent noise to induce
 * @param numCycles How many full size wave cycles required. All the data
 *        points will be fit into these cycles.
 * @param normalize Whether to normalise the data. This may be required for some
 *        layers like LSTM. Default is true.
 */
void GenerateNoisySinRNN(arma::cube& data,
                         arma::cube& labels,
                         size_t rho,
                         size_t outputSteps = 1,
                         const int dataPoints = 100,
                         const double gain = 1.0,
                         const int freq = 10,
                         const double phase = 0,
                         const int noisePercent = 20,
                         const double numCycles = 6.0,
                         const bool normalize = true)
{
  int points = dataPoints;
  int r = dataPoints % rho;

  if (r == 0)
  {
    points += outputSteps;
  }
  else
  {
    points += rho - r + outputSteps;
  }

  arma::colvec x(points);
  int i = 0;
  double interval = numCycles / freq / points;

  x.for_each([&i, gain, freq, phase, noisePercent, interval]
    (arma::colvec::elem_type& val) {
    double t = interval * (++i);
    val = gain * ::sin(2 * M_PI * freq * t + phase) +
        (noisePercent * gain / 100 * Random(0.0, 0.1));
  });

  arma::colvec y = x;
  if (normalize)
    y = arma::normalise(x);

  // Now break this into columns of rho size slices.
  size_t numColumns = y.n_elem / rho;
  data = arma::cube(1, numColumns, rho);
  labels = arma::cube(outputSteps, numColumns, 1);

  for (size_t i = 0; i < numColumns; ++i)
  {
    data.tube(0, i) = y.rows(i * rho, i * rho + rho - 1);
    labels.subcube(0, i, 0, outputSteps - 1, i, 0) =
        y.rows(i * rho + rho, i * rho + rho + outputSteps - 1);
  }
}

/**
 * @brief RNNSineTest Test a simple RNN using noisy sine. Use single output
 *        for multiple inputs.
 * @param hiddenUnits No of units in the hiddenlayer.
 * @param rho The input sequence length.
 * @param numEpochs The number of epochs to run.
 * @return The mean squared error of the prediction.
 */
double RNNSineTest(size_t hiddenUnits, size_t rho, size_t numEpochs = 100)
{
  RNN<MeanSquaredError> net(rho, true);
  net.Add<LinearNoBias>(hiddenUnits);
  net.Add<LSTM>(hiddenUnits);
  net.Add<LinearNoBias>(1);

  // Generate data
  arma::cube data;
  arma::cube labels;
  GenerateNoisySinRNN(data, labels, rho, 1, 2000, 20.0, 200, 0.0, 45, 20);

  // Break into training and test sets. Simply split along columns.
  size_t trainCols = data.n_cols * 0.8; // Take 20% out for testing.
  size_t testCols = data.n_cols - trainCols;
  arma::cube testData = data.subcube(0, data.n_cols - testCols, 0,
      data.n_rows - 1, data.n_cols - 1, data.n_slices - 1);
  arma::cube testLabels = labels.subcube(0, labels.n_cols - testCols, 0,
      labels.n_rows - 1, labels.n_cols - 1, labels.n_slices - 1);

  RMSProp opt(0.005, 16, 0.9, 1e-08, trainCols * numEpochs, 1e-5);

  net.Train(data.subcube(0, 0, 0, data.n_rows - 1, trainCols - 1,
      data.n_slices - 1), labels.subcube(0, 0, 0, labels.n_rows - 1,
      trainCols - 1, labels.n_slices - 1), opt);

  // Well now it should be trained. Do the test here.
  arma::cube prediction;
  net.Predict(testData, prediction);

  // The prediction must really follow the test data. So convert both the test
  // data and the pediction to vectors and compare the two.
  arma::colvec testVector = arma::vectorise(testData);
  arma::colvec predVector = arma::vectorise(prediction);

  // Adjust the vectors for comparison, as the prediction is one step ahead.
  testVector = testVector.rows(1, testVector.n_rows - 1);
  predVector = predVector.rows(0, predVector.n_rows - 2);
  double error = std::sqrt(arma::sum(arma::square(testVector - predVector))) /
      testVector.n_rows;

  return error;
}

/**
 * Test RNN using multiple timestep input and single output.
 */
TEST_CASE("MultiTimestepTest", "[RecurrentNetworkTest]")
{
  double err = RNNSineTest(4, 10, 20);
  REQUIRE(err <= 0.025);
}

/**
 * Test that RNN::Train() does not give an error for large rho.
 */
TEST_CASE("LargeRhoValueRnnTest", "[RecurrentNetworkTest]")
{
  // Setting rho value greater than sequence length which is 17.
  const size_t rho = 100;
  const size_t hiddenSize = 128;
  const size_t numLetters = 256;
  using MatType = arma::cube;
  std::vector<std::string>trainingData = { "THIS IS THE INPUT 0" ,
                                           "THIS IS THE INPUT 1" ,
                                           "THIS IS THE INPUT 3"};


  RNN<> model(rho);
  model.Add<LSTM>(hiddenSize);
  model.Add<Dropout>(0.1);
  model.Add<Linear>(numLetters);

  const auto makeInput = [numLetters](const char *line) -> MatType
  {
    const auto strLen = strlen(line);
    // Rows: number of dimensions.
    // Cols: number of sequences/points.
    // Slices: number of steps in sequences.
    MatType result(numLetters, 1, strLen, arma::fill::zeros);
    for (size_t i = 0; i < strLen; ++i)
    {
      result.at(static_cast<arma::uword>(line[i]), 0, i) = 1.0;
    }
    return result;
  };

  const auto makeTarget = [] (const char *line) -> MatType
  {
    const auto strLen = strlen(line);
    // Responses for NegativeLogLikelihood should be
    // non-one-hot-encoded class IDs (from 0 to num_classes - 1).
    MatType result(1, 1, strLen, arma::fill::zeros);
    // The response is the *next* letter in the sequence.
    for (size_t i = 0; i < strLen - 1; ++i)
    {
      result.at(0, 0, i) = static_cast<arma::uword>(line[i + 1]);
    }
    // The final response is empty, so we set it to class 0.
    result.at(0, 0, strLen - 1) = 0.0;
    return result;
  };

  std::vector<MatType> inputs(trainingData.size());
  std::vector<MatType> targets(trainingData.size());
  for (size_t i = 0; i < trainingData.size(); ++i)
  {
    inputs[i] = makeInput(trainingData[i].c_str());
    targets[i] = makeTarget(trainingData[i].c_str());
  }
  ens::StandardSGD opt(0.01, 1, 100);
  model.Train(inputs[0], targets[0], opt);
  INFO("Training over");
}

/**
 * Test that a simple RNN with no recurrent components behaves the same as an
 * FFN.
 */
TEST_CASE("RNNFFNTest", "[RecurrentNetworkTest]")
{
  // We'll create an RNN with *no* BPTT, just a simple single-layer linear
  // network.
  RNN<MeanSquaredError, ConstInitialization> rnn;
  FFN<MeanSquaredError, ConstInitialization> ffn;

  rnn.Add<Linear>(10);
  rnn.Add<Sigmoid>();
  rnn.Add<Linear>(1);

  ffn.Add<Linear>(10);
  ffn.Add<Sigmoid>();
  ffn.Add<Linear>(1);

  // Now create some random data.
  arma::cube data(20, 200, 1, arma::fill::randu);
  arma::cube responses(1, 200, 1, arma::fill::randu);

  // Train the FFN.
  ens::StandardSGD optimizer(1e-5, 1, 200, 1e-8, false);

  ffn.Train(data.slice(0), responses.slice(0), optimizer);
  rnn.Train(data, responses, optimizer);

  // Now, the weights should be the same!
  CheckMatrices(ffn.Parameters(), rnn.Parameters());
}
