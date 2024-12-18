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
  arma::colvec x = arma::linspace<arma::colvec>(0, points - 1, points) /
      points * 20.0;
  arma::colvec y1 = arma::sin(x + randu() * 3.0);
  arma::colvec y2 = arma::sin(x / 2.0 + randu() * 3.0);

  data = arma::zeros(1 /* single dimension */, sequences * 2, points);
  labels = arma::zeros(2 /* 2 classes */, sequences * 2);

  for (size_t seq = 0; seq < sequences; seq++)
  {
    arma::vec sequence = randu(points) * noise + y1 + (randu() - 0.5) * noise;
    for (size_t i = 0; i < points; ++i)
      data(0, seq, i) = sequence[i];

    labels(0, seq) = 1;

    sequence = randu(points) * noise + y2 + (randu() - 0.5) * noise;
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
 * Ensure LinearRecurrent networks work with larger batch sizes.
 */
TEST_CASE("LinearRecurrentBatchSizeTest", "[RecurrentNetworkTest]")
{
  BatchSizeTest<LinearRecurrent>();
}

/**
 * Ensure LSTMs work with larger batch sizes.
 */
TEST_CASE("LSTMBatchSizeTest", "[RecurrentNetworkTest]")
{
  BatchSizeTest<LSTM>();
}

/**
 * Generate a super simple impulse whose response is a step function at the same
 * time step.  The impulse occurs at a random time in each dimension.
 *
 * Predicting this sequence is a super easy task for a recurrent network, but
 * not possible without a recurrent connection.
 */
void GenerateImpulseStepData(arma::cube& data,
                             arma::cube& responses,
                             const size_t dimensions,
                             const size_t numSequences,
                             const size_t seqLen)
{
  data.zeros(dimensions, numSequences, seqLen);
  responses.zeros(dimensions, numSequences, seqLen);

  for (size_t i = 0; i < numSequences; ++i)
  {
    for (size_t j = 0; j < dimensions; ++j)
    {
      const size_t impulseStep = RandInt(0, seqLen - 1);

      data(j, i, impulseStep) = 1.0;
      responses.subcube(j, i, impulseStep, j, i, seqLen - 1).fill(1.0);
    }
  }
}

/**
 * Test that the recurrent layer is always able to learn to hold the output at 1
 * when the input impulse happens.
 */
template<typename RecurrentLayerType>
double ImpulseStepDataTest(const size_t dimensions, const size_t rho)
{
  arma::cube data, responses;

  GenerateImpulseStepData(data, responses, dimensions, 1000, 50);

  arma::cube trainData = data.cols(0, 699);
  arma::cube trainResponses = responses.cols(0, 699);
  arma::cube testData = data.cols(700, 999);
  arma::cube testResponses = responses.cols(700, 999);

  RNN<MeanSquaredError, ConstInitialization> net(rho);
  net.Add<RecurrentLayerType>(dimensions);

  const size_t numEpochs = 50;
  RMSProp opt(0.003, 32, 0.9, 1e-08, 700 * numEpochs, 1e-5);

  net.Train(trainData, trainResponses, opt, ens::ProgressBar());
  net.Parameters().print("network parameters");

  arma::cube testPreds;
  net.Predict(testData, testPreds);

  arma::rowvec testData1 = vectorise(testData.col(0)).t();
  arma::rowvec testPred1 = vectorise(testPreds.col(0)).t();
  arma::rowvec testResp1 = vectorise(testResponses.col(0)).t();

  testData1.print("testData1");
  testPred1.print("testPred1");
  testResp1.print("testResp1");

  // Compute the MSE of the test data.
  const double error = std::sqrt(sum(square(
      vectorise(testPreds) - vectorise(testResponses)))) / testPreds.n_elem;

  return error;
}

TEST_CASE("RNNImpulseStepLinearRecurrentTest", "[RecurrentNetworkTest]")
{
  const double err = ImpulseStepDataTest<LinearRecurrent>(1, 5);
  REQUIRE(err <= 0.001);
}

TEST_CASE("RNNImpulseStepLSTMTest", "[RecurrentNetworkTest]")
{
  const double err = ImpulseStepDataTest<LSTM>(1, 5);
  REQUIRE(err <= 0.001);
}

/**
 * Generates noisy sine wave into arma::cubes that can be used with `RNN`.
 *
 * @param data Will hold the generated data.
 * @param responses Will hold the generated responses (data shifted by one time
 *    step).
 * @param numSequences Number of sequences to generate.
 * @param seqLen Length of each sequence (time steps).
 * @param gain The maximum gain on the amplitude
 * @param freq The maximum frequency of the sine wave
 * @param noisePercent The percent noise to induce
 * @param numCycles How many full size wave cycles required. All the data
 *        points will be fit into these cycles.
 */
void GenerateNoisySinRNN(arma::cube& data,
                         arma::cube& responses,
                         const size_t numSequences,
                         const size_t seqLen,
                         const double gain = 1.0,
                         const double freq = 0.05,
                         const double noisePercent = 5)
{
  data.set_size(1, numSequences, seqLen + 1);

  for (size_t i = 0; i < numSequences; ++i)
  {
    // Create the time steps with a random offset.
    arma::vec t = 100.0 * Random() + arma::linspace<arma::vec>(0, seqLen,
        seqLen + 1);

    data.tube(0, i) = gain * (arma::sin(2 * M_PI * freq * t)); //+
         //noisePercent / 100.0 * arma::randu<arma::vec>(seqLen + 1));
  }

  // Make the responses as a time-shifted version of the data.
  responses = data.slices(1, data.n_slices - 1);
  data.shed_slice(data.n_slices - 1);
}

/**
 * @brief RNNSineTest Test a simple RNN using noisy sine. Use single output
 *        for multiple inputs.
 * @param hiddenUnits No of units in the hiddenlayer.
 * @param rho The input sequence length.
 * @param numEpochs The number of epochs to run.
 * @return The mean squared error of the prediction.
 */
template<typename LayerType>
double RNNSineTest(size_t hiddenUnits, size_t rho, size_t numEpochs = 10)
{
  RNN<MeanSquaredError> net(rho);
  net.Add<LinearNoBias>(hiddenUnits);
  net.Add<LeakyReLU>();
  net.Add<LayerType>(hiddenUnits);
  net.Add<LeakyReLU>();
  net.Add<LinearNoBias>(1);

  // Generate data
  arma::cube data, responses;
  GenerateNoisySinRNN(data, responses, 500, rho + 10);

  arma::colvec dataV = vectorise(data.col(0));
  arma::colvec respV = vectorise(responses.col(0));
  std::cout << "data: " << dataV.rows(0, 10).t();
  std::cout << "resp: " << respV.rows(0, 10).t();

  // Break into training and test sets. Simply split along columns.
  size_t trainCols = data.n_cols * 0.8; // Take 20% out for testing.
  size_t testCols = data.n_cols - trainCols;
  arma::cube testData = data.subcube(0, data.n_cols - testCols, 0,
      data.n_rows - 1, data.n_cols - 1, data.n_slices - 1);
  arma::cube testResponses = responses.subcube(0, responses.n_cols - testCols,
      0, responses.n_rows - 1, responses.n_cols - 1, responses.n_slices - 1);

  RMSProp opt(0.001, 500, 0.9, 1e-08, trainCols * numEpochs, 1e-5);

  net.Train(data.subcube(0, 0, 0, data.n_rows - 1, trainCols - 1,
      data.n_slices - 1), responses.subcube(0, 0, 0, responses.n_rows - 1,
      trainCols - 1, responses.n_slices - 1), opt, ens::ProgressBar());

  // Well now it should be trained. Do the test here.
  arma::cube prediction;
  net.Predict(testData, prediction);

  // The prediction must really follow the test data. So convert both the test
  // data and the pediction to vectors and compare the two.
  arma::colvec testVector = vectorise(testData.col(0));
  arma::colvec predVector = vectorise(prediction.col(0));
  arma::colvec testResp = vectorise(testResponses.col(0));

  std::cout << "testVector: " << testVector.rows(0, 10).t();
  std::cout << "predVector: " << predVector.rows(0, 10).t();
  std::cout << "testResp:   " << testResp.rows(0, 10).t();

  // Adjust the vectors for comparison, as the prediction is one step ahead.
  testVector = testVector.rows(1, testVector.n_rows - 1);
  predVector = predVector.rows(0, predVector.n_rows - 2);
  double error = std::sqrt(sum(square(testVector - predVector))) /
      testVector.n_rows;

  return error;
}

/**
 * Test RNN using multiple timestep input and single output.
 */
TEST_CASE("RNNSineLinearRecurrentTest", "[RecurrentNetworkTest]")
{
  const double err2 = RNNSineTest<Linear>(5, 15, 20);
  std::cout << "err2: " << err2 << "\n";
  const double err = RNNSineTest<LinearRecurrent>(5, 15, 20);
  std::cout << "err: " << err << "\n";
  REQUIRE(err <= 0.05);
}

TEST_CASE("RNNSineLSTMTest", "[RecurrentNetworkTest]")
{
  const double err = RNNSineTest<LSTM>(4, 10, 20);
  std::cout << "err: " << err << "\n";
  REQUIRE(err <= 0.25);
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
  std::vector<std::string> trainingData = {
      "test input string 1",
      "other test input string",
      "more test input I like it",
      "test input string 2",
      "test input string 10",
      "another test input string" };

  size_t maxStrLen = 0;
  for (size_t i = 0; i < trainingData.size(); ++i)
  {
    maxStrLen = std::max(maxStrLen, trainingData[i].size());
  }

  // Assemble the data into a one-hot encoded cube.  Note that for
  // NegativeLogLikelihood, the output should be a categorical.
  arma::cube data(numLetters, trainingData.size(), maxStrLen);
  arma::cube outputs(1, trainingData.size(), maxStrLen);
  for (size_t i = 0; i < trainingData.size(); ++i)
  {
    for (size_t j = 0; j < trainingData[i].size() - 1; ++j)
    {
      const size_t c = (size_t) trainingData[i][j];
      data(c, i, j) = 1.0;
      outputs(0, i, j) = (size_t) trainingData[i][j + 1];
    }

    const size_t c = (size_t) trainingData[i][trainingData[i].size() - 1];
    data(c, i, trainingData[i].size() - 1) = 1.0;
    outputs(0, i, trainingData[i].size() - 1) = 255; // signifies end of string
  }

  // Now build the model.
  RNN<> model(rho);
  model.Add<LinearRecurrent>(hiddenSize);
  model.Add<LeakyReLU>();
  model.Add<Linear>(numLetters);
  model.Add<LogSoftMax>();

  // Train the model and ensure that it gives reasonable results.
  // Use a very small learning rate to prevent divergence on this problem.
  ens::StandardSGD opt(1e-15, 16, 1 * data.n_cols /* 1 epoch */);
  model.Train(data, outputs, opt);

  // Ensure that none of the weights are NaNs or Inf.
  REQUIRE(!model.Parameters().has_nan());
  REQUIRE(!model.Parameters().has_inf());
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
  ens::StandardSGD optimizer(1e-5, 100, 200, 1e-8, false);

  ffn.Train(data.slice(0), responses.slice(0), optimizer);
  rnn.Train(data, responses, optimizer);

  // Now, the weights should be the same!
  CheckMatrices(ffn.Parameters(), rnn.Parameters());
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
 * @param output The generated output sequence.
 * @param numSequences The number of sequences to generate.
 */
void GenerateDistractedSequence(arma::cube& input,
                                arma::cube& output,
                                const size_t numSequences)
{
  input.zeros(10, numSequences, 10);
  output.zeros(2, numSequences, 10);

  for (size_t i = 0; i < numSequences; ++i)
  {
    arma::uvec index = arma::shuffle(arma::linspace<arma::uvec>(0, 7, 8));

    // Set the target in the input sequence and the corresponding targets in the
    // output sequence by following the correct order.
    const size_t idx = RandInt(0, 2);
    input(idx, i, index(0)) = 1;
    // The response for this index comes first if index(j) comes before the
    // other target index.
    output(idx, i, (index(0) > index(1)) ? 9 : 8) = 1;

    const size_t idx2 = (idx + 1) % 2;
    input(idx2, i, index(1)) = 1;
    output(idx2, i, (index(1) > index(0)) ? 9 : 8) = 1;

    for (size_t j = 2; j < 8; ++j)
      input(2 + RandInt(0, 6), i, index(j)) = 1;

    // Set the prompts which direct the network to give an answer.
    input(8, i, 8) = 1;
    input(9, i, 9) = 1;
  }
}

// This custom ensmallen callback that computes the number of sequences
// that are predicted correctly.  If that goes above a threshold, we terminate
// early.
class DistractedSequenceTestSetCallback
{
 public:
  DistractedSequenceTestSetCallback(const arma::cube& testInput,
                                    const arma::cube& testLabels) :
      testInput(testInput), testLabels(testLabels), error(1.0) { }

  // This is called at the end of each epoch of training.
  template<typename OptimizerType, typename FunctionType, typename MatType>
  bool EndEpoch(OptimizerType& /* opt */,
                FunctionType& network,
                const MatType& /* coordinates */,
                const size_t epoch,
                const double objective)
  {
    // Don't bother checking accuracy before 50 epochs.
    if (epoch < 50)
      return false;

    // Compute the predictions on the test set.
    arma::cube testPreds;
    network.Predict(testInput, testPreds);

    // Binarize the output to 0/1.
    for (size_t j = 0; j < testPreds.n_slices; ++j)
      data::Binarize(testPreds.slice(j), testPreds.slice(j), 0.5);

    // Count the number of columns where we got one or more time slice
    // predictions incorrect.
    //
    // The expression is a little complicated, but the inner max(sum(...))
    // returns 1 if a sequence was wrong and 0 if a sequence was correct.
    const double numIncorrect = accu(max(max(testLabels != testPreds, 0), 2));
    error = std::min(error, numIncorrect / testLabels.n_cols);

    std::cout << "Epoch " << epoch << ": error " << error << "; objective "
        << objective << ".\n";
    if (error <= 0.15)
    {
      // Terminate the optimization early.
      return true;
    }

    // Continue the optimization.
    return false;
  }

  // Get the last computed error.
  double Error() const { return error; }

 private:
  const arma::cube& testInput;
  const arma::cube& testLabels;
  double error;
};

/**
 * Train the specified network and the construct distracted sequence recall
 * dataset.
 */
template<typename RecurrentLayerType>
void DistractedSequenceRecallTestNetwork(
    const size_t cellSize, const size_t hiddenSize)
{
  arma::cube trainInput, trainLabels, testInput, testLabels;
  const size_t trainDistractedSequenceCount = 1000;
  const size_t testDistractedSequenceCount = 1000;

  // Generate the training and test data.
  GenerateDistractedSequence(trainInput, trainLabels,
      trainDistractedSequenceCount);
  GenerateDistractedSequence(testInput, testLabels,
      testDistractedSequenceCount);

  // Construct a simple network with 10 input units, a recurrent layer, a hidden
  // layer, then a sigmoid for the output (so all outputs are between 0 and 1).
  const size_t outputSize = 2;
  const size_t rho = 10;

  // Initialize weights in [-0.1, 0.1) as suggested in the paper.
  RNN<MeanSquaredError> model(rho, false, MeanSquaredError(),
      RandomInitialization(-0.1, 0.1));
  model.Add<LinearNoBias>(cellSize);
  model.Add<RecurrentLayerType>(hiddenSize);
  model.Add<LinearNoBias>(outputSize);
  model.Add<Sigmoid>();

  // Make a forward pass to initialize the weights.  Then we will set the biases
  // to 0.
  arma::cube tmp;
  model.Predict(trainInput, tmp);
  //(dynamic_cast<LSTM*>(model.Network()[1]))->InputGateBias().zeros();
  //(dynamic_cast<LSTM*>(model.Network()[1]))->OutputGateBias().zeros();
  //(dynamic_cast<LSTM*>(model.Network()[1]))->ForgetGateBias().zeros();
  //(dynamic_cast<LSTM*>(model.Network()[1]))->BlockInputBias().zeros();

  // Allow up to 250 epochs for training.  In the paper, the standard LSTM took
  // on average 80k iterations (so 80 epochs, since we have 1k training
  // sequences) before the network reached 95% accuracy.
  StandardSGD opt(0.015, 16, 250 * trainInput.n_cols, 1e-8);

  // This callback will terminate training early when accuracy reaches 90%.  At
  // least 50 epochs of training are required.
  DistractedSequenceTestSetCallback cb(testInput, testLabels);
  model.Train(trainInput, trainLabels, opt, cb);
  std::cout << "Parameter size: " << model.Parameters().n_elem << "\n";

  // We only require 85% accuracy.
  REQUIRE(cb.Error() <= 0.15);
}

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
TEST_CASE("LSTMDistractedSequenceRecallTest", "[RecurrentNetworkTest]")
{
  DistractedSequenceRecallTestNetwork<LSTM>(10, 8);
}
