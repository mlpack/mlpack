/**
 * @file tests/split_data_test.cpp
 * @author Tham Ngap Wei
 *
 * Test the SplitData method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "test_catch_tools.hpp"
#include "catch.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

/**
 * Compare the data after train test split.  This assumes that the labels
 * correspond to each column, so that we can easily check each point against its
 * original.
 *
 * @param inputData The original data set before split.
 * @param compareData The data want to compare with the inputData;
 *   it could be train data or test data.
 * @param inputLabel The labels of each point in compareData.
 */
void CompareData(const mat& inputData,
                 const mat& compareData,
                 const Row<size_t>& inputLabel)
{
  for (size_t i = 0; i != compareData.n_cols; ++i)
  {
    const mat& lhsCol = inputData.col(inputLabel(i));
    const mat& rhsCol = compareData.col(i);
    for (size_t j = 0; j != lhsCol.n_rows; ++j)
    {
      if (std::abs(rhsCol(j)) < 1e-5)
        REQUIRE(lhsCol(j) == Approx(0.0).margin(1e-5));
      else
        REQUIRE(lhsCol(j) == Approx(rhsCol(j)).epsilon(1e-7));
    }
  }
}

void CheckMatEqual(const mat& inputData,
                   const mat& compareData)
{
  const mat& sortedInput = arma::sort(inputData, "ascend", 1);
  const mat& sortedCompare = arma::sort(compareData, "ascend", 1);
  for (size_t i = 0; i < sortedInput.n_cols; ++i)
  {
    const mat& lhsCol = sortedInput.col(i);
    const mat& rhsCol = sortedCompare.col(i);
    for (size_t j = 0; j < lhsCol.n_rows; ++j)
    {
      if (std::abs(rhsCol(j)) < 1e-5)
        REQUIRE(lhsCol(j) == Approx(0.0).margin(1e-5));
      else
        REQUIRE(lhsCol(j) == Approx(rhsCol(j)).epsilon(1e-7));
    }
  }
}

/**
 * Check that no labels have been duplicated.
 */
void CheckDuplication(const Row<size_t>& trainLabels,
                      const Row<size_t>& testLabels)
{
  // Assemble a vector that will hold the counts of each element.
  Row<size_t> counts(trainLabels.n_elem + testLabels.n_elem);
  counts.zeros();

  for (size_t i = 0; i < trainLabels.n_elem; ++i)
  {
    REQUIRE(trainLabels[i] < counts.n_elem);
    counts[trainLabels[i]]++;
  }
  for (size_t i = 0; i < testLabels.n_elem; ++i)
  {
    REQUIRE(testLabels[i] < counts.n_elem);
    counts[testLabels[i]]++;
  }

  // Now make sure each point has been used once.
  for (size_t i = 0; i < counts.n_elem; ++i)
    REQUIRE(counts[i] == 1);
}

TEST_CASE("SplitShuffleDataResultMat", "[SplitDataTest]")
{
  mat input(2, 10);
  input = reshape(linspace(0, 19, 20), 2, 10);

  mat trainData, testData;
  Split(input, trainData, testData, 0.2);
  REQUIRE(trainData.n_cols == 8);
  REQUIRE(testData.n_cols == 2);

  mat concat = join_rows(trainData, testData);
  CheckMatEqual(input, concat);
}

TEST_CASE("SplitDataResultMat", "[SplitDataTest]")
{
  mat input(2, 10);
  input = reshape(linspace(0, 19, 20), 2, 10);

  mat trainData, testData;
  Split(input, trainData, testData, 0.2, false);
  REQUIRE(trainData.n_cols == 8); // Train data.
  REQUIRE(testData.n_cols == 2); // Test data.

  mat concat = join_rows(trainData, testData);
  // Order matters here.
  CheckMatrices(input, concat);
}

TEST_CASE("ZeroRatioSplitData", "[SplitDataTest]")
{
  mat input(2, 10);
  input = reshape(linspace(0, 19, 20), 2, 10);

  mat trainData, testData;
  Split(input, trainData, testData, 0, false);
  REQUIRE(trainData.n_cols == 10); // Train data.
  REQUIRE(testData.n_cols == 0); // Test data.

  mat concat = join_rows(trainData, testData);
  // Order matters here.
  CheckMatrices(input, concat);
}

TEST_CASE("TotalRatioSplitData", "[SplitDataTest]")
{
  mat input(2, 10);
  input = reshape(linspace(0, 19, 20), 2, 10);

  mat trainData, testData;
  Split(input, trainData, testData, 1, false);
  REQUIRE(trainData.n_cols == 0); // Train data.
  REQUIRE(testData.n_cols == 10); // Test data.

  mat concat = join_rows(trainData, testData);
  // Order matters here.
  CheckMatrices(input, concat);
}

TEST_CASE("SplitLabeledDataResultMat", "[SplitDataTest]")
{
  mat input(2, 10);
  input.randu();

  // Set the labels to the column ID, so that CompareData can compare the data
  // after Split is called.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;
  Split(input, labels, trainData, testData, trainLabels, testLabels, 0.2);
  REQUIRE(trainData.n_cols == 8);
  REQUIRE(testData.n_cols == 2);
  REQUIRE(trainLabels.n_cols == 8);
  REQUIRE(testLabels.n_cols == 2);

  CompareData(input, trainData, trainLabels);
  CompareData(input, testData, testLabels);

  // The last thing to check is that we aren't duplicating any points in the
  // train or test labels.
  CheckDuplication(trainLabels, testLabels);
}

TEST_CASE("SplitCheckSize", "[SplitDataTest]")
{
  mat input = randu<mat>(2, 10);

  const Row<size_t> firstLabels = arma::linspace<Row<size_t>>(0,
      input.n_cols - 1, input.n_cols);

  const Row<size_t> secondLabels = arma::linspace<Row<size_t>>(0,
      input.n_cols, input.n_cols + 1);

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;

  REQUIRE_THROWS_AS(Split(input, secondLabels, trainData, testData, trainLabels,
      testLabels, 0.2), std::invalid_argument);

  REQUIRE_NOTHROW(Split(input, firstLabels, trainData, testData, trainLabels,
      testLabels, 0.2));
}

/**
 * The same test as above, but on a larger dataset.
 */
TEST_CASE("SplitDataLargerTest", "[SplitDataTest]")
{
  mat input(10, 497);
  input = reshape(linspace(0, 4969, 4970), 10, 497);

  mat trainData, testData;
  Split(input, trainData, testData, 0.3);
  REQUIRE(trainData.n_cols == 497 - size_t(0.3 * 497));
  REQUIRE(testData.n_cols == size_t(0.3 * 497));

  mat concat = join_rows(trainData, testData);
  CheckMatEqual(input, concat);
}

TEST_CASE("SplitLabeledDataLargerTest", "[SplitDataTest]")
{
  mat input(10, 497);
  input.randu();

  // Set the labels to the column ID.
  const Row<size_t> labels = arma::linspace<Row<size_t>>(0, input.n_cols - 1,
      input.n_cols);

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;
  Split(input, labels, trainData, testData, trainLabels, testLabels, 0.3);
  REQUIRE(trainData.n_cols == 497 - size_t(0.3 * 497));
  REQUIRE(testData.n_cols == size_t(0.3 * 497));
  REQUIRE(trainLabels.n_cols == 497 - size_t(0.3 * 497));
  REQUIRE(testLabels.n_cols == size_t(0.3 * 497));

  CompareData(input, trainData, trainLabels);
  CompareData(input, testData, testLabels);

  CheckDuplication(trainLabels, testLabels);
}

/**
 * Check that test ratio of 0 results in a full train set for stratified split.
 */
TEST_CASE("ZeroRatioStratifiedSplitData", "[SplitDataTest]")
{
  mat input(2, 15);
  input.randu();

  // Set the labels to 5 0s and 10 1s.
  const Row<size_t> labels = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  const double testRatio = 0;

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;
  StratifiedSplit(input, labels, trainData, testData, trainLabels, testLabels,
      testRatio, false);
  REQUIRE(trainData.n_cols == 15);
  REQUIRE(testData.n_cols == 0);
  REQUIRE(trainLabels.n_cols == 15);
  REQUIRE(testLabels.n_cols == 0);
}

/**
 * Check that test ratio of 1 results in a full test set for stratified split.
 */
TEST_CASE("TotalRatioStratifiedSplitData", "[SplitDataTest]")
{
  mat input(2, 15);
  input.randu();

  // Set the labels to 5 0s and 10 1s.
  const Row<size_t> labels = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  const double testRatio = 1;

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;

  StratifiedSplit(input, labels, trainData, testData, trainLabels, testLabels,
      testRatio, false);
  REQUIRE(trainData.n_cols == 0);
  REQUIRE(testData.n_cols == 15);
  REQUIRE(trainLabels.n_cols == 0);
  REQUIRE(testLabels.n_cols == 15);
}

/**
 * Check if data is stratified according to labels.
 */
TEST_CASE("StratifiedSplitDataResultTest", "[SplitDataTest]")
{
  mat input(5, 24);
  input.randu();

  // Set the labels to 4 0s, 8 1s and 12 2s.
  const Row<size_t> labels = { 0, 0, 0, 0,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
  const double testRatio = 0.25;

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;

  StratifiedSplit(input, labels, trainData, testData, trainLabels, testLabels,
      testRatio, true);
  REQUIRE(static_cast<uvec>(find(trainLabels == 0)).n_rows == 3);
  REQUIRE(static_cast<uvec>(find(trainLabels == 1)).n_rows == 6);
  REQUIRE(static_cast<uvec>(find(trainLabels == 2)).n_rows == 9);

  REQUIRE(static_cast<uvec>(find(testLabels == 0)).n_rows == 1);
  REQUIRE(static_cast<uvec>(find(testLabels == 1)).n_rows == 2);
  REQUIRE(static_cast<uvec>(find(testLabels == 2)).n_rows == 3);

  mat concat = join_rows(trainData, testData);
  CheckMatEqual(input, concat);
}

/**
 * Check if data is stratified according to labels on a larger data set.
 * Example calculation to find resultant number of samples in the train and
 * test set:
 *
 * Since there are 256 0s and the test ratio is 0.3,
 * Number of 0s in the test set = 76 ( floor(256 * 0.3) = floor(76.8) ).
 * Number of 0s in the train set = 180 ( 256 - 76 ).
 */
TEST_CASE("StratifiedSplitLargerDataResultTest", "[SplitDataTest]")
{
  mat input(3, 480);
  input.randu();

  // 256 0s, 128 1s, 64 2s and 32 3s.
  Row<size_t> zeroLabel(256);
  Row<size_t> oneLabel(128);
  Row<size_t> twoLabel(64);
  Row<size_t> threeLabel(32);

  zeroLabel.fill(0);
  oneLabel.fill(1);
  twoLabel.fill(2);
  threeLabel.fill(3);

  Row<size_t> labels = join_rows(zeroLabel, oneLabel);
  labels = join_rows(labels, twoLabel);
  labels = join_rows(labels, threeLabel);
  const double testRatio = 0.3;

  mat trainData, testData;
  Row<size_t> trainLabels, testLabels;
  StratifiedSplit(input, labels, trainData, testData, trainLabels, testLabels,
      testRatio, false);
  REQUIRE(static_cast<uvec>(find(trainLabels == 0)).n_rows == 180);
  REQUIRE(static_cast<uvec>(find(trainLabels == 1)).n_rows == 90);
  REQUIRE(static_cast<uvec>(find(trainLabels == 2)).n_rows == 45);
  REQUIRE(static_cast<uvec>(find(trainLabels == 3)).n_rows == 23);

  REQUIRE(static_cast<uvec>(find(testLabels == 0)).n_rows == 76);
  REQUIRE(static_cast<uvec>(find(testLabels == 1)).n_rows == 38);
  REQUIRE(static_cast<uvec>(find(testLabels == 2)).n_rows == 19);
  REQUIRE(static_cast<uvec>(find(testLabels == 3)).n_rows == 9);

  mat concat = join_rows(trainData, testData);
  CheckMatEqual(input, concat);
}

/**
 * Check that Split() with stratifyData true throws a runtime error if labels
 * are not of type arma::Row<>.
 */
TEST_CASE("StratifiedSplitRunTimeErrorTest", "[SplitDataTest]")
{
  mat input(3, 480);
  mat labels(2, 480);
  input.randu();
  labels.randu();

  const double testRatio = 0.3;

  mat trainData, testData, trainLabels, testLabels;

  REQUIRE_THROWS_AS(StratifiedSplit(input, labels, trainData, testData,
      trainLabels, testLabels, testRatio, false), std::runtime_error);
}

/*
 * Split with input of type field<mat>.
 */
TEST_CASE("SplitDataResultField", "[SplitDataTest]")
{
  field<mat> input(1, 2);

  mat matA(2, 10);
  mat matB(2, 10);

  matA = linspace(0, matA.n_elem - 1);
  matB = linspace(matA.n_elem, matA.n_elem + matB.n_elem - 1);

  input(0, 0) = matA;
  input(0, 1) = matB;

  field<mat> trainData, testData;

  Split(input, trainData, testData, 0.5, false);
  REQUIRE(trainData.n_cols == 1); // Train data.
  REQUIRE(testData.n_cols == 1); // Test data.

  field<mat> concat = {trainData(0), testData(0)};
  // Order matters here.
  CheckFields(input, concat);
}

/**
 * Test for Split() with labels of type arma::Mat with shuffleData = False.
 */
TEST_CASE("SplitMatrixLabeledData", "[SplitDataTest]")
{
  const mat input(2, 10, fill::randu);
  const mat labels(2, 10, fill::randu);

  mat trainData, testData, trainLabels, testLabels;

  Split(input, labels, trainData, testData, trainLabels, testLabels, 0.2,
      false);
  REQUIRE(trainData.n_cols == 8);
  REQUIRE(testData.n_cols == 2);
  REQUIRE(trainLabels.n_cols == 8);
  REQUIRE(testLabels.n_cols == 2);

  mat inputConcat = join_rows(trainData, testData);
  mat labelsConcat = join_rows(trainLabels, testLabels);

  // Order matters here.
  CheckMatrices(input, inputConcat);
  CheckMatrices(labels, labelsConcat);
}

/**
 * Split with input of type field<mat> and label of type field<vec>.
 */
TEST_CASE("SplitLabeledDataResultField", "[SplitDataTest]")
{
  field<mat> input(1, 2);
  field<vec> label(1, 2);

  mat matA(2, 10, fill::randu);
  mat matB(2, 10, fill::randu);

  vec vecA(10, fill::randu);
  vec vecB(10, fill::randu);

  input(0, 0) = matA;
  input(0, 1) = matB;

  label(0, 0) = vecA;
  label(0, 1) = vecB;

  field<mat> trainData, testData;
  field<vec> trainLabels, testLabels;

  Split(input, label, trainData, testData, trainLabels, testLabels, 0.5, false);
  REQUIRE(trainData.n_cols == 1);   // Train data.
  REQUIRE(testData.n_cols == 1);    // Test data.
  REQUIRE(trainLabels.n_cols == 1); // Train label.
  REQUIRE(testLabels.n_cols == 1);  // Test label.

  field<mat> inputConcat = {trainData(0), testData(0)};
  field<vec> labelConcat = {trainLabels(0), testLabels(0)};

  // Order matters here.
  CheckFields(input, inputConcat);
  CheckFields(label, labelConcat);
}

/**
 * Split arma::cube data without labels.
 */
TEST_CASE("SplitCubeData", "[SplitDataTest]")
{
  cube input(10, 30, 5, fill::randu);

  cube trainInput, testInput;

  Split(input, trainInput, testInput, 0.1, false);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainInput.n_slices == 5);
  REQUIRE(approx_equal(trainInput, input.cols(0, 26), "both", 1e-5, 1e-5));

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testInput.n_slices == 5);
  REQUIRE(approx_equal(testInput, input.cols(27, 29), "both", 1e-5, 1e-5));
}

/**
 * Split arma::cube data without labels, and shuffle it.
 */
TEST_CASE("SplitCubeDataShuffle", "[SplitDataTest]")
{
  cube input(10, 30, 5, fill::randu);

  cube trainInput, testInput;

  Split(input, trainInput, testInput, 0.1, true);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainInput.n_slices == 5);

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testInput.n_slices == 5);

  // Make sure we can find each column of the data.
  std::vector<bool> found(30, false);
  for (size_t i = 0; i < trainInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(trainInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(testInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);
}

/**
 * Split arma::cube data with labels but without shuffling.
 */
TEST_CASE("SplitCubeDataWithLabels", "[SplitDataTest]")
{
  // These have the same shape that might be used for RNNs.
  cube input(10, 30, 5, fill::randu);
  cube labels(1, 30, 5, fill::randu);

  cube trainInput, trainLabels, testInput, testLabels;

  Split(input, labels, trainInput, testInput, trainLabels, testLabels, 0.1,
      false);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainInput.n_slices == 5);
  REQUIRE(approx_equal(trainInput, input.cols(0, 26), "both", 1e-5, 1e-5));

  REQUIRE(trainLabels.n_rows == 1);
  REQUIRE(trainLabels.n_cols == 27);
  REQUIRE(trainLabels.n_slices == 5);
  REQUIRE(approx_equal(trainLabels, labels.cols(0, 26), "both", 1e-5, 1e-5));

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testInput.n_slices == 5);
  REQUIRE(approx_equal(testInput, input.cols(27, 29), "both", 1e-5, 1e-5));

  REQUIRE(testLabels.n_rows == 1);
  REQUIRE(testLabels.n_cols == 3);
  REQUIRE(testLabels.n_slices == 5);
  REQUIRE(approx_equal(testLabels, labels.cols(27, 29), "both", 1e-5, 1e-5));
}

/**
 * Split arma::cube data with labels and with shuffling.
 */
TEST_CASE("SplitCubeDataShuffleWithLabels", "[SplitDataTest]")
{
  // These have the same shape that might be used for RNNs.
  cube input(10, 30, 5, fill::randu);
  cube labels(1, 30, 5, fill::randu);

  cube trainInput, trainLabels, testInput, testLabels;

  Split(input, labels, trainInput, testInput, trainLabels, testLabels, 0.1,
      false);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainInput.n_slices == 5);

  REQUIRE(trainLabels.n_rows == 1);
  REQUIRE(trainLabels.n_cols == 27);
  REQUIRE(trainLabels.n_slices == 5);

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testInput.n_slices == 5);

  REQUIRE(testLabels.n_rows == 1);
  REQUIRE(testLabels.n_cols == 3);
  REQUIRE(testLabels.n_slices == 5);

  // Make sure we can find each column of the data.
  std::vector<bool> found(30, false);
  for (size_t i = 0; i < trainInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(trainInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(testInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);

  // Make sure we can find each column of the labels.
  found.flip();
  for (size_t i = 0; i < trainLabels.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(trainLabels.col(i), labels.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(testLabels.col(i), labels.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);
}

/**
 * Split arma::cube data with labels and weights, but without shuffling.
 */
TEST_CASE("SplitCubeDataWithLabelsAndWeights", "[SplitDataTest]")
{
  // These have the same shape that might be used for RNNs.
  cube input(10, 30, 5, fill::randu);
  cube labels(1, 30, 5, fill::randu);
  // Just for fun, set the weights to have a different type.
  frowvec weights(30, fill::randu);

  cube trainInput, trainLabels, testInput, testLabels;
  frowvec trainWeights, testWeights;

  Split(input, labels, weights, trainInput, testInput, trainLabels, testLabels,
      trainWeights, testWeights, 0.1, false);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainInput.n_slices == 5);
  REQUIRE(approx_equal(trainInput, input.cols(0, 26), "both", 1e-5, 1e-5));

  REQUIRE(trainLabels.n_rows == 1);
  REQUIRE(trainLabels.n_cols == 27);
  REQUIRE(trainLabels.n_slices == 5);
  REQUIRE(approx_equal(trainLabels, labels.cols(0, 26), "both", 1e-5, 1e-5));

  REQUIRE(trainWeights.n_elem == 27);
  REQUIRE(approx_equal(trainWeights, weights.subvec(0, 26), "both", 1e-5,
      1e-5));

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testInput.n_slices == 5);
  REQUIRE(approx_equal(testInput, input.cols(27, 29), "both", 1e-5, 1e-5));

  REQUIRE(testLabels.n_rows == 1);
  REQUIRE(testLabels.n_cols == 3);
  REQUIRE(testLabels.n_slices == 5);
  REQUIRE(approx_equal(testLabels, labels.cols(27, 29), "both", 1e-5, 1e-5));

  REQUIRE(testWeights.n_elem == 3);
  REQUIRE(approx_equal(testWeights, weights.subvec(27, 29), "both", 1e-5,
      1e-5));
}

/**
 * Split arma::cube data with labels and weights, and also shuffle it while
 * splitting.
 */
TEST_CASE("SplitCubeDataShuffleWithLabelsAndWeights", "[SplitDataTest]")
{
  // These have the same shape that might be used for RNNs.
  cube input(10, 30, 5, fill::randu);
  cube labels(1, 30, 5, fill::randu);
  // Just for fun, set the weights to have a different type.
  frowvec weights(30, fill::randu);

  cube trainInput, trainLabels, testInput, testLabels;
  frowvec trainWeights, testWeights;

  Split(input, labels, weights, trainInput, testInput, trainLabels, testLabels,
      trainWeights, testWeights, 0.1, true);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainInput.n_slices == 5);

  REQUIRE(trainLabels.n_rows == 1);
  REQUIRE(trainLabels.n_cols == 27);
  REQUIRE(trainLabels.n_slices == 5);

  REQUIRE(trainWeights.n_elem == 27);

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testInput.n_slices == 5);

  REQUIRE(testLabels.n_rows == 1);
  REQUIRE(testLabels.n_cols == 3);
  REQUIRE(testLabels.n_slices == 5);

  REQUIRE(testWeights.n_elem == 3);

  // Make sure we can find each column of the data.
  std::vector<bool> found(30, false);
  for (size_t i = 0; i < trainInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(trainInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(testInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);

  // Make sure we can find each column of the labels.
  found.flip();
  for (size_t i = 0; i < trainLabels.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(trainLabels.col(i), labels.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testLabels.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(testLabels.col(i), labels.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);

  // Make sure we can find each weight.
  found.flip();
  for (size_t i = 0; i < trainWeights.n_elem; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (trainWeights[i] == weights[c])
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testWeights.n_elem; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (testWeights[i] == weights[c])
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);
}

/**
 * Test that we can split regular matrix data with labels and weights.
 */
TEST_CASE("SplitMatDataShuffleWithLabelsAndWeights", "[SplitDataTest]")
{
  mat input(10, 30, fill::randu);
  Row<size_t> labels = linspace<Row<size_t>>(10, 39, 30);
  frowvec weights(30, fill::randu);

  mat trainInput, testInput;
  Row<size_t> trainLabels, testLabels;
  frowvec trainWeights, testWeights;

  Split(input, labels, weights, trainInput, testInput, trainLabels, testLabels,
      trainWeights, testWeights, 0.1, true);

  REQUIRE(trainInput.n_rows == 10);
  REQUIRE(trainInput.n_cols == 27);
  REQUIRE(trainLabels.n_elem == 27);
  REQUIRE(trainWeights.n_elem == 27);

  REQUIRE(testInput.n_rows == 10);
  REQUIRE(testInput.n_cols == 3);
  REQUIRE(testLabels.n_elem == 3);
  REQUIRE(testWeights.n_elem == 3);

  // Make sure we can find each column of the data.
  std::vector<bool> found(30, false);
  for (size_t i = 0; i < trainInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(trainInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testInput.n_cols; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (approx_equal(testInput.col(i), input.col(c), "both", 1e-5, 1e-5))
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);

  // Make sure we can find each column of the labels.
  found.flip();
  for (size_t i = 0; i < trainLabels.n_elem; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (trainLabels[i] == labels[c])
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testLabels.n_elem; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (testLabels[i] == labels[c])
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);

  // Make sure we can find each weight.
  found.flip();
  for (size_t i = 0; i < trainWeights.n_elem; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (trainWeights[i] == weights[c])
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t i = 0; i < testWeights.n_elem; ++i)
  {
    for (size_t c = 0; c < 30; ++c)
    {
      if (testWeights[i] == weights[c])
      {
        REQUIRE(found[c] == false);
        found[c] = true;
      }
    }
  }

  for (size_t c = 0; c < 30; ++c)
    REQUIRE(found[c]);
}
