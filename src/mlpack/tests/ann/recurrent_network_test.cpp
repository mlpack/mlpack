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
#define ENS_PRINT_WARN
#define ENS_PRINT_INFO
#include <mlpack/core.hpp>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack/methods/ann/ann.hpp>

#include "../catch.hpp"
#include "../serialization.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;
using namespace ens;

TEST_CASE("RNNImpulseStepLinearRecurrentTest", "[RecurrentNetworkTest][long]")
{
  double err = ImpulseStepDataTest<LinearRecurrent<>>(1, 5);
  REQUIRE(err <= 0.001);

  err = ImpulseStepDataTest<LinearRecurrent<>>(3, 5);
  REQUIRE(err <= 0.003);

  err = ImpulseStepDataTest<LinearRecurrent<>>(5, 5);
  REQUIRE(err <= 0.005);
}

TEST_CASE("RNNImpulseStepLSTMTest", "[RecurrentNetworkTest][long]")
{
  double err = ImpulseStepDataTest<LSTM<>>(1, 5);
  REQUIRE(err <= 0.001);

  err = ImpulseStepDataTest<LSTM<>>(3, 5);
  REQUIRE(err <= 0.001);

  err = ImpulseStepDataTest<LSTM<>>(5, 5);
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

    data.tube(0, i) = gain * (arma::sin(2 * M_PI * freq * t) +
         noisePercent / 100.0 * arma::randu<arma::vec>(seqLen + 1));
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
  net.Add<LayerType>(hiddenUnits);
  net.Add<Linear>(1);

  // Generate data
  arma::cube data, responses;
  GenerateNoisySinRNN(data, responses, 500, rho + 10);

  arma::colvec dataV = vectorise(data.col(0));
  arma::colvec respV = vectorise(responses.col(0));

  // Break into training and test sets. Simply split along columns.
  size_t trainCols = data.n_cols * 0.8; // Take 20% out for testing.
  size_t testCols = data.n_cols - trainCols;
  arma::cube testData = data.subcube(0, data.n_cols - testCols, 0,
      data.n_rows - 1, data.n_cols - 1, data.n_slices - 1);
  arma::cube testResponses = responses.subcube(0, responses.n_cols - testCols,
      0, responses.n_rows - 1, responses.n_cols - 1, responses.n_slices - 1);

  RMSProp opt(0.001, 1, 0.99, 1e-08, trainCols * numEpochs, 1e-5);

  net.Train(data.subcube(0, 0, 0, data.n_rows - 1, trainCols - 1,
      data.n_slices - 1), responses.subcube(0, 0, 0, responses.n_rows - 1,
      trainCols - 1, responses.n_slices - 1), opt);

  // Well now it should be trained. Do the test here.
  arma::cube prediction;
  net.Predict(testData, prediction);

  // The prediction must really follow the test data. So convert both the test
  // data and the pediction to vectors and compare the two.
  arma::colvec testVector = vectorise(testData.col(0));
  arma::colvec predVector = vectorise(prediction.col(0));
  arma::colvec testResp = vectorise(testResponses.col(0));

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
TEMPLATE_TEST_CASE("RNNSineTest", "[RecurrentNetworkTest][long]",
    LinearRecurrent<>,
    LSTM<>,
    GRU<>)
{
  // This can sometimes fail due to bad initializations or bad luck.  So, try it
  // up to three times.
  bool success = false;
  for (size_t t = 0; t < 3; ++t)
  {
    const double err = RNNSineTest<TestType>(3, 3, 50);
    if (err <= 0.08)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Create a simple recurrent neural network for the noisy sines task, but such
 * that every point is the same; then, ensure that when we sweep the batch size,
 * the results are exactly the same.
 */
template<typename RecurrentLayerType>
void BatchSizeTest()
{
  const size_t bpttTruncate = 10;

  arma::cube onePointInput, onePointResponses;
  GenerateNoisySinRNN(onePointInput, onePointResponses, 1, bpttTruncate + 10);

  // We don't have repmat for cubes, so this is a little tedious.
  arma::cube input(onePointInput.n_rows, 500, onePointInput.n_slices);
  arma::cube responses(onePointResponses.n_rows, 500,
      onePointResponses.n_slices);
  for (size_t i = 0; i < onePointInput.n_slices; ++i)
  {
    input.slice(i) = repmat(onePointInput.slice(i), 1, 500);
    responses.slice(i) = repmat(onePointResponses.slice(i), 1, 500);
  }

  RNN<MeanSquaredError> model(bpttTruncate);
  model.Add<RecurrentLayerType>(onePointResponses.n_rows);

  model.Reset(1);
  arma::mat initParams = model.Parameters();

  // Run with a batch size of 1.
  StandardSGD opt(0.1, 1, 1);
  opt.Shuffle() = false;
  model.Train(input, responses, opt);

  arma::mat targetOutput = model.Parameters();

  // Now re-run with larger batch sizes.
  for (size_t bsPow = 1; bsPow < 6; ++bsPow)
  {
    const size_t batchSize = std::pow((size_t) 2, bsPow);

    #if ENS_VERSION_MAJOR >= 3
    opt = StandardSGD(0.1, batchSize, batchSize);
    #else
    // Older versions of ensmallen did not adjust the step size for the batch
    // size.
    opt = StandardSGD(0.1 / ((double) batchSize), batchSize, batchSize);
    #endif
    opt.Shuffle() = false;
    model.Reset(1);
    model.Parameters() = initParams;
    model.Train(input, responses, opt);

    // This is trained with one point.
    arma::mat outputParams = model.Parameters();
    REQUIRE(approx_equal(targetOutput, outputParams, "both", 1e-6, 1e-6));
  }
}

/**
 * Ensure recurrent layers work with larger batch sizes.
 */
TEMPLATE_TEST_CASE("RNNBatchSizeTest", "[RecurrentNetworkTest][tiny]",
    LinearRecurrent<>,
    LSTM<>,
    GRU<>)
{
  BatchSizeTest<TestType>();
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
  #if ENS_VERSION_MAJOR >= 3
  ens::StandardSGD opt(1.6e-14, 16, 1 * data.n_cols /* 1 epoch */);
  #else
  // Older versions of ensmallen did not adjust the step size for the batch
  // size.
  ens::StandardSGD opt(1e-15, 16, 1 * data.n_cols /* 1 epoch */);
  #endif
  model.Train(data, outputs, opt);

  // Ensure that none of the weights are NaNs or Inf.
  REQUIRE(!model.Parameters().has_nan());
  REQUIRE(!model.Parameters().has_inf());
}

/**
 * Test that a simple RNN with no recurrent components behaves the same as an
 * FFN.
 */
TEST_CASE("RNNFFNTest", "[RecurrentNetworkTest][tiny]")
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
  #if ENS_VERSION_MAJOR >= 3
  ens::StandardSGD optimizer(1e-3, 100, 200, 1e-8, false);
  #else
  // Older versions of ensmallen did not adjust the step size for the batch
  // size.
  ens::StandardSGD optimizer(1e-5, 100, 200, 1e-8, false);
  #endif

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
                const double /* objective */)
  {
    // Don't bother checking accuracy before 5 epochs.
    if (epoch < 5)
      return false;

    // Compute the predictions on the test set.
    arma::cube testPreds;
    network.Predict(testInput, testPreds);

    // Binarize the output to 0/1.
    for (size_t j = 0; j < testPreds.n_slices; ++j)
      Binarize(testPreds.slice(j), testPreds.slice(j), 0.5);

    // Count the number of columns where we got one or more time slice
    // predictions incorrect.
    //
    // The expression is a little complicated, but the inner max(sum(...))
    // returns 1 if a sequence was wrong and 0 if a sequence was correct.
    const double numIncorrect = accu(max(max(testLabels != testPreds, 0), 2));
    error = std::min(error, numIncorrect / testLabels.n_cols);
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
  model.Add<Linear>(cellSize);
  model.Add<RecurrentLayerType>(hiddenSize);
  model.Add<Linear>(outputSize);
  model.Add<Sigmoid>();

  // Make a forward pass to initialize the weights.  Then we will set the biases
  // to 0.
  arma::cube tmp;
  model.Predict(trainInput, tmp);

  // Allow up to 250 epochs for training.  In the paper, the standard LSTM took
  // on average 80k iterations (so 80 epochs, since we have 1k training
  // sequences) before the network reached 95% accuracy.
  Adam opt(0.008, 8, 0.9, 0.999, 1e-8, 250 * trainInput.n_cols, 1e-8);

  // This callback will terminate training early when accuracy reaches 85%.  At
  // least 50 epochs of training are required.
  DistractedSequenceTestSetCallback cb(testInput, testLabels);
  model.Train(trainInput, trainLabels, opt, cb);

  // If the error is greater than 15%, we allow a few restarts.
  size_t trial = 0;
  while (cb.Error() > 0.15 && trial < 3)
  {
    opt = Adam(0.008, 8, 0.9, 0.999, 1e-8, 250 * trainInput.n_cols, 1e-8);

    model.Reset();
    model.Train(trainInput, trainLabels, opt, cb);

    ++trial;
  }

  // We only require 85% accuracy.
  REQUIRE(cb.Error() <= 0.15);
}

/**
 * Train the specified networks on the Derek D. Monner's distracted sequence
 * recall task.
 */
TEST_CASE("LSTMDistractedSequenceRecallTest", "[RecurrentNetworkTest][long]")
{
  DistractedSequenceRecallTestNetwork<LSTM<>>(10, 8);
}

/**
 * Construct a 2-class dataset out of noisy sines.  Each class corresponds to a
 * different frequency of sine wave.
 *
 * @param data Input data used to store the noisy sines.
 * @param labels Labels used to store the target class of the noisy sines.
 * @param points Number of points/features in a single sequence.
 * @param sequences Number of sequences for each class.
 * @param noise The noise factor that influences the sines.
 */
void GenerateNoisySines(arma::cube& data,
                        arma::cube& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.05)
{
  data.zeros(1 /* single dimension */, sequences, points);
  labels.zeros(1 /* will hold 2 classes */, sequences, points);

  for (size_t seq = 0; seq < sequences; seq++)
  {
    const size_t label = (seq % 2);
    const double freq = (label == 0) ? 1 : 0.25;

    // Each sequence will be either a quarter or a sixteenth of a sine wave.
    const arma::vec t = arma::linspace<arma::vec>(0, points - 1, points) /
        points * 0.25 + Random();

    data.tube(0, seq) = arma::sin(2 * M_PI * t * freq) +
        noise * arma::randn<arma::vec>(points);
    labels.tube(0, seq).fill(label);
  }
}

/**
 * Train a simple RNN to perform a classification task.
 */
TEST_CASE("SequenceClassificationTest", "[RecurrentNetworkTest]")
{
  // It isn't guaranteed that the recurrent network will converge in the
  // specified number of iterations using random weights. If this works 1 of 5
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;
  const size_t rho = 10;

  for (size_t trial = 0; trial < 5; ++trial)
  {
    // Generate 500 (2 * 250) noisy sines. A single sine contains rho
    // points/features.
    arma::cube input;
    arma::cube labels;
    GenerateNoisySines(input, labels, rho + 5, 500);

    // Construct a very simple network.
    RNN<> model(rho);
    model.Add<LSTM>(3);
    model.Add<Linear>(2);
    model.Add<LogSoftMax>();

    StandardSGD opt(0.005, 16, 15 * input.n_cols, -100);
    model.Train(input, labels, opt);

    arma::cube predictions;
    model.Predict(input, predictions);

    size_t error = 0;
    for (size_t i = 0; i < predictions.n_cols; ++i)
    {
      const size_t predictedClass =
          predictions.slice(predictions.n_slices - 1).col(i).index_max();
      const size_t targetClass = labels(0, i, labels.n_slices - 1);

      if (predictedClass != targetClass)
        error++;
    }

    const double accuracy = 1.0 - double(error) / predictions.n_cols;
    if (accuracy >= 0.8)
    {
      ++successes;
      break;
    }
  }

  REQUIRE(successes >= 1);
}

/**
 * Train a simple RNN to perform a classification task, but in 'single' mode, so
 * the response is only backpropagated at the final time step.
 */
TEST_CASE("SequenceClassificationSingleTest", "[RecurrentNetworkTest][long]")
{
  size_t successes = 0;
  const size_t rho = 10;

  for (size_t trial = 0; trial < 3; ++trial)
  {
    // Generate 500 (2 * 250) noisy sines. A single sine contains rho
    // points/features.
    arma::cube input;
    arma::cube labels;
    GenerateNoisySines(input, labels, rho, 500);

    // For single model, strip the labels down to just one slice.
    labels.shed_slices(1, labels.n_slices - 1);

    // Construct a very simple network.
    RNN<> model(rho, true /* single response mode */);
    model.Add<LSTM>(3);
    model.Add<Linear>(2);
    model.Add<LogSoftMax>();

    // Note that we increase the learning rate over the non-single mode, and
    // increase the number of epochs, since it can take longer for BPTT to
    // converge with only one error signal.
    StandardSGD opt(0.1, 1, 30 * input.n_cols, -100);
    model.Train(input, labels, opt);

    arma::cube predictions;
    model.Predict(input, predictions);

    size_t error = 0;
    for (size_t i = 0; i < predictions.n_cols; ++i)
    {
      const size_t predictedClass = predictions.slice(0).col(i).index_max();
      const size_t targetClass = labels(0, i, 0);

      if (predictedClass != targetClass)
        error++;
    }

    const double accuracy = 1.0 - double(error) / predictions.n_cols;
    // We allow a little more leeway than the non-single test, because the
    // problem here is a bit harder.
    if (accuracy >= 0.7)
    {
      ++successes;
      break;
    }
  }

  REQUIRE(successes > 0);
}

/**
 * Make sure the RNN can be properly serialized.
 */
TEST_CASE("RNNSerializationTest", "[RecurrentNetworkTest]")
{
  const size_t rho = 10;

  // Generate 12 (2 * 6) noisy sines. A single sine contains rho
  // points/features.
  arma::cube input;
  arma::cube labels;
  GenerateNoisySines(input, labels, rho, 6);

  RNN<> model(rho);
  model.Add<LSTM>(3);
  model.Add<LinearRecurrent>(3);
  model.Add<GRU>(3);
  model.Add<Linear>(2);
  model.Add<LogSoftMax>();

  StandardSGD opt(0.001, 1, input.n_cols /* 1 epoch */, -100);
  model.Train(input, labels, opt);

  // Serialize the network.
  RNN<> xmlModel(1), jsonModel(3), binaryModel(5);
  SerializeObjectAll(model, xmlModel, jsonModel, binaryModel);

  // Take predictions, check the output.
  arma::cube prediction, xmlPrediction, jsonPrediction, binaryPrediction;
  model.Predict(input, prediction);
  xmlModel.Predict(input, xmlPrediction);
  jsonModel.Predict(input, jsonPrediction);
  binaryModel.Predict(input, binaryPrediction);

  CheckMatrices(prediction, xmlPrediction, jsonPrediction, binaryPrediction);
}

//
// Reber grammar tests: a series of tests for RNNs that ensure that a simple
// grammar (that can be expressed as a state machine) can be learned by LSTMs.
// The Reber grammar task (and the embedded Reber grammar task) appear in many
// LSTM papers, including the original.
//

// Return the Reber state transition matrix.  The row indicates the internal
// state.  From every state, there are two possible transitions.  The letter
// associated with each transition is encoded in the first two columns; the
// numeric values of the corresponding states are in the second two columns.
// next path).
inline arma::Mat<char> ReberTransitionMatrix()
{
  return arma::Mat<char>({{ 'T', 'P', '1', '2' },
                          { 'X', 'S', '3', '1' },
                          { 'V', 'T', '4', '2' },
                          { 'X', 'S', '2', '5' },
                          { 'P', 'V', '3', '5' },
                          { 'E', 'E', '0', '0' }});
}

// Return a map from Reber grammar characters to indices.
std::unordered_map<char, size_t> ReberToDimMap()
{
  std::unordered_map<char, size_t> charDimMap;
  charDimMap['B'] = 0;
  charDimMap['T'] = 1;
  charDimMap['S'] = 2;
  charDimMap['X'] = 3;
  charDimMap['P'] = 4;
  charDimMap['V'] = 5;
  charDimMap['E'] = 6;

  return charDimMap;
}

// Return a map from indices to Reber grammar character.
std::unordered_map<size_t, char> DimToReberMap()
{
  std::unordered_map<size_t, char> dimCharMap;
  dimCharMap[0] = 'B';
  dimCharMap[1] = 'T';
  dimCharMap[2] = 'S';
  dimCharMap[3] = 'X';
  dimCharMap[4] = 'P';
  dimCharMap[5] = 'V';
  dimCharMap[6] = 'E';

  return dimCharMap;
}

// Generate a string from the Reber grammar.
inline std::string GenerateReberString(const bool embedded = false)
{
  const arma::Mat<char> transitions = ReberTransitionMatrix();

  std::string result = "B";
  size_t state = 0;
  while (result.back() != 'E')
  {
    const size_t choice = RandInt(0, 2);
    result += transitions(state, choice);
    state = (transitions(state, choice + 2) - '0');
  }

  if (embedded)
  {
    const size_t dir = RandInt(0, 2);
    if (dir == 0)
      result = "BP" + result + "PE";
    else
      result = "BT" + result + "TE";
  }

  return result;
}

// Given the input sequence `input`, check that every character in response
// satisfies the Reber grammar transition.
inline bool IsReberResponse(const std::string& input,
                            const std::string& response,
                            const bool embedded = false)
{
  const arma::Mat<char> transitions = ReberTransitionMatrix();

  // If we are embedded, we have to check the first and last characters
  // separately.
  size_t startOffset = 0;
  size_t endOffset = 0;
  if (embedded)
  {
    startOffset = 2;
    endOffset = 2;

    // The first response must be T or P.
    if (response[0] != 'T' && response[0] != 'P')
      return false;
    // The second response must be a B.
    if (response[1] != 'B')
      return false;

    // The second-to-last response must be the same as the first input.
    if (response[response.size() - 2] != input[1])
      return false;
    // The last response must be an E.
    if (response[response.size() - 1] != 'E')
      return false;
  }

  // Check the regular Reber grammar part of the string (which may or may not be
  // the whole string, depending on the value of `embedded`).
  size_t state = 0;
  for (size_t i = startOffset; i < input.size() - 1 - endOffset; ++i)
  {
    if (response[i] != transitions(state, 0) &&
        response[i] != transitions(state, 1))
      return false;

    if (input[i + 1] == transitions(state, 0))
      state = (transitions(state, 2) - '0');
    else if (input[i + 1] == transitions(state, 1))
      state = (transitions(state, 3) - '0');
    else
      return false;
  }

  return true;
}

// Convert a prediction cube back into a string.  This expects prediction.n_rows
// to be 7, and prediction.n_cols to be 1.
inline std::string PredictionToReberString(const arma::cube& prediction)
{
  const std::unordered_map<size_t, char> dimCharMap = DimToReberMap();

  std::string result = "";
  for (size_t s = 0; s < prediction.n_slices; ++s)
    result += dimCharMap.at(prediction.slice(s).index_max());

  return result;
}

/**
 * Train the specified network and the construct a Reber grammar dataset.
 */
template<typename ModelType>
void ReberGrammarTestNetwork(ModelType& model,
                             const bool embedded = false,
                             const size_t maxEpochs = 30,
                             const size_t trials = 5)
{
  const size_t trainSize = 1000;
  const size_t testSize = 1000;

  // Each input sequence might have a different length, so, we have to use an
  // arma::field, and we will train on each sequence with a separate call to
  // Train().
  arma::field<arma::cube> trainInput, trainLabels, testInput;
  const std::unordered_map<char, size_t> charDimMap = ReberToDimMap();

  // Generate the training data.
  trainInput.set_size(trainSize);
  trainLabels.set_size(trainSize);
  for (size_t i = 0; i < trainSize; ++i)
  {
    const std::string reber = GenerateReberString(embedded);

    trainInput[i].zeros(7, 1, reber.length() - 1);
    trainLabels[i].zeros(7, 1, reber.length() - 1);

    for (size_t j = 0; j < reber.length() - 1; ++j)
    {
      trainInput[i](charDimMap.at(reber[j]), 0, j) = 1.0;
      trainLabels[i](charDimMap.at(reber[j + 1]), 0, j) = 1.0;
    }
  }

  // Generate the test data (responses are not needed to check that the
  // predictions are valid).
  testInput.set_size(testSize);
  for (size_t i = 0; i < testSize; ++i)
  {
    const std::string reber = GenerateReberString(embedded);

    testInput[i].zeros(7, 1, reber.length() - 1);
    for (size_t j = 0; j < reber.length() - 1; ++j)
      testInput[i](charDimMap.at(reber[j]), 0, j) = 1.0;
  }

  // It isn't guaranteed that the recurrent network will converge in the
  // specified number of iterations using random weights. If this works 1 of 5
  // times, I'm fine with that. All I want to know is that the network is able
  // to escape from local minima and to solve the task.
  size_t successes = 0;
  double error = 0.0;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    // Reset model before using for next trial.
    model.Reset(trainInput[0].n_rows);
    // This will only run one iteration for one grammar.
    Adam opt(embedded ? 0.05 : 0.1, 1, 0.9, 0.999, 1e-8, 1);
    opt.ResetPolicy() = false;

    for (size_t epoch = 0; epoch < maxEpochs; epoch++)
    {
      double loss = 0.0;
      for (size_t j = 0; j < trainSize; ++j)
      {
        // Each input sequence may have a different length, so we need to train
        // them differently.
        model.BPTTSteps() = trainInput[j].n_slices;
        loss += model.Train(trainInput[j], trainLabels[j], opt);
      }


      // Ask the network to predict the next Reber grammar in the given
      // sequence.
      error = 0.0;
      for (size_t i = 0; i < testSize; ++i)
      {
        arma::cube prediction;
        model.Predict(testInput[i], prediction);

        const std::string inputString = PredictionToReberString(testInput[i]);
        const std::string predictedString = PredictionToReberString(prediction);
        if (!IsReberResponse(inputString, predictedString, embedded))
          ++error;
      }

      error /= testSize;
      // If the error is less than 30%, terminate early.
      if (error <= 0.3)
      {
        ++successes;
        break;
      }
    }

    if (successes > 0)
      break;
  }

  REQUIRE(successes >= 1);
}

TEST_CASE("LSTMReberGrammarTest", "[RecurrentNetworkTest][tiny]")
{
  // Note that our performance doesn't exactly match the LSTM paper that
  // originally introduced this task.  But, part of this is probably that they
  // did some really specific things for initialization that we don't here.
  RNN<MeanSquaredError> model(5, false, MeanSquaredError(),
      RandomInitialization(-0.5, 0.5));
  model.Add<LSTM>(4);
  model.Add<Linear>(7);
  model.Add<Sigmoid>();
  ReberGrammarTestNetwork(model, false);
}

/**
 * Train the specified networks on an embedded Reber grammar dataset.
 */
TEMPLATE_TEST_CASE("RNNEmbeddedReberGrammarTest",
    "[RecurrentNetworkTest][long]",
    LSTM<>,
    GRU<>)
{
  RNN<MeanSquaredError> model(5, false, MeanSquaredError(),
      RandomInitialization(-0.1, 0.1));
  // Sometimes a few extra units are needed to effectively get the embedded
  // Reber grammar every time.
  model.Add<TestType>(25);
  model.Add<Linear>(7);
  model.Add<Sigmoid>();
  ReberGrammarTestNetwork(model, true);
}

/**
 * Test that we can train an RNN on sequences of different lengths, and get
 * roughly the same thing we would for training on non-ragged sequences.
 */
TEMPLATE_TEST_CASE("RNNRaggedSequenceTest", "[RecurrentNetworkTest][long]",
    LSTM<>,
    GRU<>)
{
  const size_t rho = 25;
  const size_t numEpochs = 3;

  #if ENS_VERSION_MAJOR >= 3
  const double stepSize = 0.024;
  #else
  // Older versions of ensmallen did not adjust the step size for the batch
  // size.
  const double stepSize = 0.003;
  #endif

  // Generate noisy sine data.
  arma::cube data, responses;
  GenerateNoisySinRNN(data, responses, 500, rho + 35);
  arma::cube origData = data;
  arma::cube origResponses = responses;

  // Assign random sequence lengths for each sine.
  arma::urowvec lengths = arma::randi<arma::urowvec>(500, DistrParam(40, 60));

  // Set garbage data for anything past the end of a sequence.
  for (size_t c = 0; c < 500; ++c)
  {
    if (lengths[c] == 60)
      continue;

    data.subcube(0, c, lengths[c],
                 data.n_rows - 1, c, data.n_slices - 1).randu();
    responses.subcube(0, c, lengths[c],
                      responses.n_rows - 1, c, responses.n_slices - 1).randu();
  }

  bool success = false;
  // Try the test up to 5 times since the random initalization can cause some
  // differences between the two networks.
  for (size_t attempt = 0; attempt < 5; attempt++)
  {
    // Build a network and train it.
    RMSProp opt(stepSize, 8, 0.99, 1e-08, 500 * numEpochs, 1e-5);

    RNN<MeanSquaredError> net(rho);
    net.Add<TestType>(10);
    net.Add<Linear>(1);

    // Train on all the data.
    net.Train(data, responses, lengths, opt);

    // Make sure that the predictions match the data reasonably.
    arma::cube prediction;
    net.Predict(data, prediction, lengths);

    // Sum the error for all sequences.
    size_t timeSteps = 0;
    double totalError = 0.0;
    for (size_t c = 0; c < 500; ++c)
    {
      timeSteps += lengths[c];
      totalError += accu(abs(vectorise(responses.subcube(
          0, c, 0, responses.n_rows - 1, c, lengths[c] - 1)) -
          vectorise(prediction.subcube(
          0, c, 0, prediction.n_rows - 1, c, lengths[c] - 1))));
    }

    const double averageError = (totalError / timeSteps);

    // Now compute another network where we don't use the sequence lengths.
    RNN<MeanSquaredError> net2(rho);
    net2.Add<TestType>(10);
    net2.Add<Linear>(1);

    // Train and predict, then compute the sum error.
    RMSProp opt2(stepSize, 8, 0.99, 1e-08, 500 * numEpochs / 2, 1e-5);
    net2.Train(origData, origResponses, opt2);
    net2.Predict(origData, prediction);
    const double refAverageError = mean(abs(vectorise(origResponses) -
        vectorise(prediction)));

    // There can be some margin in the results because we are not training on as
    // much data for the ragged sequences.
    success = abs(refAverageError - averageError) <= 0.1;
    if (success)
      break;
  }

  REQUIRE(success);
}

/**
 * Test that we can train a RecurrentLinear RNN on sequences of different
 * lengths, and get roughly the same thing we would for training on non-ragged
 * sequences. This is a separate test from `RNNRaggedSequenceTest` because the
 * `LinearRecurrent` layer can't model sine waves.
 */
TEST_CASE("RNNRecurrentLinearRaggedSequenceTest",
    "[RecurrentNetworkTest][long]")
{
  const size_t rho = 10;
  const size_t numEpochs = 10;

  #if ENS_VERSION_MAJOR >= 3
  const double stepSize = 0.08;
  #else
  // Older versions of ensmallen did not adjust the step size for the batch
  // size.
  const double stepSize = 0.01;
  #endif

  // Generate data.
  // The response is equal to the sum of the predictors.
  // eg. for predictors: 0, 5, 3, 1,  3
  //          responses: 0, 5, 8, 9, 12
  arma::cube predictors, responses;
  arma::urowvec sequenceLengths;

  predictors.randn(1, 500, rho + 20);
  responses.zeros(1, 500, rho + 20);
  for (size_t c = 0; c < 500; c++)
  {
    for (size_t t = 0; t < responses.n_slices; t++)
    {
      responses(0, c, t) = arma::accu(predictors.subcube(0, c, 0, 0, c, t));
    }
  }

  // Assign random sequence lengths for each sequence.
  arma::urowvec lengths = arma::randi<arma::urowvec>(500,
      arma::distr_param(10, 30));

  arma::cube origPredictors = predictors;
  arma::cube origResponses = responses;

  // Set garbage data for anything past the end of a sequence.
  for (size_t c = 0; c < 500; c++)
  {
    if (lengths[c] == 30)
      continue;

    predictors.subcube(0, c, lengths[c],
                       0, c, predictors.n_slices - 1).randn();
    responses.subcube(0, c, lengths[c],
                      0, c, responses.n_slices - 1).fill(500);
  }

  bool success = false;
  // Try the test up to 5 times since the random initalization can cause some
  // differences between the two networks.
  for (size_t attempt = 0; attempt < 5; attempt++)
  {
    // Train on non-ragged sequences
    RMSProp opt1(stepSize, 8, 0.99, 1e-08, 500 * numEpochs / 2, 1e-5);
    RNN<MeanSquaredError> net1(rho);
    net1.Add<LinearRecurrent>(1);
    net1.Train(origPredictors, origResponses, opt1);

    // Test the error of the network non-ragged sequences;
    arma::cube prediction1;
    net1.Predict(origPredictors, prediction1);
    const double refAverageError = mean(abs(vectorise(origResponses) -
        vectorise(prediction1)));

    // Train on ragged sequences
    RMSProp opt2(stepSize, 8, 0.99, 1e-08, 500 * numEpochs, 1e-5);
    RNN<MeanSquaredError> net2(rho);
    net2.Add<LinearRecurrent>(1);
    net2.Train(predictors, responses, lengths, opt2);

    // Test the error of the network trained on ragged sequences;
    arma::cube prediction2;
    net2.Predict(predictors, prediction2, lengths);
    size_t timeSteps = 0;
    double totalError = 0.0;
    for (size_t c = 0; c < 500; ++c)
    {
      timeSteps += lengths[c];
      totalError += accu(abs(
          vectorise(origResponses.subcube(0, c, 0, 0, c, lengths[c] - 1)) -
          vectorise(prediction2.subcube(0, c, 0, 0, c, lengths[c] - 1))));
    }
    const double averageError = (totalError / timeSteps);

    // There can be some margin in the results because we are not training on as
    // much data for the ragged sequences.
    success = abs(refAverageError - averageError) <= 0.1;
    if (success)
      break;
  }

  REQUIRE(success);
}

/**
 * Test that `RNN::Evaluate()` returns a reasonable value.
 */
TEST_CASE("RNNEvaluateTest", "[RecurrentNetworkTest]")
{
  arma::cube predictors, labels;
  predictors.zeros(1, 5, 10);
  labels.ones(1, 5, 10);

  // Create the network.
  RNN<MeanSquaredError> net(0, false, MeanSquaredError(true));
  net.Add<LinearNoBias>(1);

  // The input is all zeros so the output should also be all zeros.
  // The loss at each slice should then be equal to the mean squared error
  // between 0 (output) and 1 (labels) which is 1.
  double loss = net.Evaluate(predictors, labels);

  // The total loss should be equal to the total number of slices.
  REQUIRE(loss == Approx(labels.n_cols * labels.n_slices));
}

/**
 * Test that `RNN::Evaluate()` returns a reasonable value with a ragged input
 * sequence.
 */
TEST_CASE("RNNRaggedEvaluateTest", "[RecurrentNetworkTest]")
{
  arma::cube predictors, labels;
  predictors.zeros(1, 5, 10);
  labels.ones(1, 5, 10);

  // Create sequence lengths from 2 to 10.
  arma::urowvec lengths(5);
  for (int i = 0; i < 5; i++)
    lengths[i] = (i + 1) * 2;

  // Create the network.
  RNN<MeanSquaredError> net(0, false, MeanSquaredError(true));
  net.Add<LinearNoBias>(1);

  // The input is all zeros so the output should also be all zeros.
  // The loss at each slice should then be equal to the mean squared error
  // between 0 (output) and 1 (labels) which is 1.
  double loss = net.Evaluate(predictors, labels, lengths, 4);

  // The total loss should be equal to the total number of slices.
  int totalSlices = 0;
  for (unsigned int i = 0; i < 5; i++)
    totalSlices += lengths[i];
  REQUIRE(loss == Approx(totalSlices));
}
