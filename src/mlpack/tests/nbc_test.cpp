/**
 * @file tests/nbc_test.cpp
 *
 * Test for the Naive Bayes classifier.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/naive_bayes.hpp>

#include "catch.hpp"

using namespace mlpack;

TEST_CASE("NaiveBayesClassifierTest", "[NBCTest][tiny]")
{
  const char* trainFilename = "trainSet.csv";
  const char* testFilename = "testSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  const char* testResultFilename = "testRes.csv";
  const char* testResultProbsFilename = "testResProbs.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!Load(trainResultFilename, trainRes))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbcTest(trainData, labels, classes);

  size_t dimension = nbcTest.Means().n_rows;
  calcMat.zeros(2 * dimension + 1, classes);

  for (size_t i = 0; i < dimension; ++i)
  {
    for (size_t j = 0; j < classes; ++j)
    {
      calcMat(i, j) = nbcTest.Means()(i, j);
      calcMat(i + dimension, j) = nbcTest.Variances()(i, j);
    }
  }

  for (size_t i = 0; i < classes; ++i)
    calcMat(2 * dimension, i) = nbcTest.Probabilities()(i);

  for (size_t i = 0; i < calcMat.n_rows; ++i)
  {
    for (size_t j = 0; j < classes; ++j)
    {
      REQUIRE(trainRes(i, j) + .00001 ==
          Approx(calcMat(i, j)).epsilon(0.0001));
    }
  }

  arma::mat testData;
  arma::Mat<size_t> testRes;
  arma::mat testResProbs;
  arma::Row<size_t> calcVec;
  arma::mat calcProbs;
  if (!Load(testFilename, testData))
    FAIL("Cannot load dataset");
  if (!Load(testResultFilename, testRes))
    FAIL("Cannot load dataset");
  if (!Load(testResultProbsFilename, testResProbs))
    FAIL("Cannot load dataset");

  testData.shed_row(testData.n_rows - 1); // Remove the labels.

  nbcTest.Classify(testData, calcVec, calcProbs);

  for (size_t i = 0; i < testData.n_cols; ++i)
    REQUIRE(testRes(i) == calcVec(i));

  for (size_t i = 0; i < testResProbs.n_cols; ++i)
  {
    for (size_t j = 0; j < testResProbs.n_rows; ++j)
    {
      REQUIRE(testResProbs(j, i) + 0.0001 ==
          Approx(calcProbs(j, i) + 0.0001).epsilon(0.0001));
    }
  }
}

// The same test, but this one uses the incremental algorithm to calculate
// variance.
TEST_CASE("NaiveBayesClassifierIncrementalTest", "[NBCTest]")
{
  const char* trainFilename = "trainSet.csv";
  const char* testFilename = "testSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  const char* testResultFilename = "testRes.csv";
  const char* testResultProbsFilename = "testResProbs.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!Load(trainResultFilename, trainRes))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbcTest(trainData, labels, classes, true);

  size_t dimension = nbcTest.Means().n_rows;
  calcMat.zeros(2 * dimension + 1, classes);

  for (size_t i = 0; i < dimension; ++i)
  {
    for (size_t j = 0; j < classes; ++j)
    {
      calcMat(i, j) = nbcTest.Means()(i, j);
      calcMat(i + dimension, j) = nbcTest.Variances()(i, j);
    }
  }

  for (size_t i = 0; i < classes; ++i)
    calcMat(2 * dimension, i) = nbcTest.Probabilities()(i);

  for (size_t i = 0; i < calcMat.n_cols; ++i)
  {
    for (size_t j = 0; j < classes; ++j)
    {
      REQUIRE(trainRes(j, i) + .00001 ==
          Approx(calcMat(j, i)).epsilon(0.0001));
    }
  }

  arma::mat testData;
  arma::Mat<size_t> testRes;
  arma::mat testResProba;
  arma::Row<size_t> calcVec;
  arma::mat calcProbs;
  if (!Load(testFilename, testData))
    FAIL("Cannot load dataset");
  if (!Load(testResultFilename, testRes))
    FAIL("Cannot load dataset");
  if (!Load(testResultProbsFilename, testResProba))
    FAIL("Cannot load dataset");

  testData.shed_row(testData.n_rows - 1); // Remove the labels.

  nbcTest.Classify(testData, calcVec, calcProbs);

  for (size_t i = 0; i < testData.n_cols; ++i)
    REQUIRE(testRes(i) == calcVec(i));

  for (size_t i = 0; i < testResProba.n_cols; ++i)
  {
    for (size_t j = 0; j < testResProba.n_rows; ++j)
    {
      REQUIRE(testResProba(j, i) + .00001 ==
          Approx(calcProbs(j, i) + .00001).epsilon(0.0001));
    }
  }
}

/**
 * Ensure that separate training gives the same model.
 */
TEST_CASE("SeparateTrainTest", "[NBCTest]")
{
  const char* trainFilename = "trainSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!Load(trainResultFilename, trainRes))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc(trainData, labels, classes, true);
  NaiveBayesClassifier<> nbcTrain(trainData.n_rows, classes);
  nbcTrain.Train(trainData, labels, classes, false);

  REQUIRE(nbc.Means().n_rows == nbcTrain.Means().n_rows);
  REQUIRE(nbc.Means().n_cols == nbcTrain.Means().n_cols);
  REQUIRE(nbc.Variances().n_rows == nbcTrain.Variances().n_rows);
  REQUIRE(nbc.Variances().n_cols == nbcTrain.Variances().n_cols);
  REQUIRE(nbc.Probabilities().n_elem ==
                      nbcTrain.Probabilities().n_elem);

  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    if (std::abs(nbc.Means()[i]) < 1e-5)
      REQUIRE(nbcTrain.Means()[i] == Approx(0.0).margin(1e-5));
    else
      REQUIRE(nbc.Means()[i] == Approx(nbcTrain.Means()[i]).epsilon(1e-7));
  }

  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    if (std::abs(nbc.Variances()[i]) < 1e-5)
      REQUIRE(nbcTrain.Variances()[i] == Approx(0.0).margin(1e-5));
    else
    {
      REQUIRE(nbc.Variances()[i] ==
          Approx(nbcTrain.Variances()[i]).epsilon(1e-7));
    }
  }

  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    if (std::abs(nbc.Probabilities()[i]) < 1e-5)
      REQUIRE(nbcTrain.Probabilities()[i] == Approx(0.0).margin(1e-5));
    else
    {
      REQUIRE(nbc.Probabilities()[i] ==
          Approx(nbcTrain.Probabilities()[i]).epsilon(1e-7));
    }
  }
}

TEST_CASE("SeparateTrainIncrementalTest", "[NBCTest]")
{
  const char* trainFilename = "trainSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!Load(trainResultFilename, trainRes))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc(trainData, labels, classes, true);
  NaiveBayesClassifier<> nbcTrain(trainData.n_rows, classes);
  nbcTrain.Train(trainData, labels, classes, true);

  REQUIRE(nbc.Means().n_rows == nbcTrain.Means().n_rows);
  REQUIRE(nbc.Means().n_cols == nbcTrain.Means().n_cols);
  REQUIRE(nbc.Variances().n_rows == nbcTrain.Variances().n_rows);
  REQUIRE(nbc.Variances().n_cols == nbcTrain.Variances().n_cols);
  REQUIRE(nbc.Probabilities().n_elem ==
                      nbcTrain.Probabilities().n_elem);

  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    if (std::abs(nbc.Means()[i]) < 1e-5)
      REQUIRE(nbcTrain.Means()[i] == Approx(0.0).margin(1e-5));
    else
      REQUIRE(nbc.Means()[i] == Approx(nbcTrain.Means()[i]).epsilon(1e-7));
  }

  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    if (std::abs(nbc.Variances()[i]) < 1e-5)
      REQUIRE(nbcTrain.Variances()[i] == Approx(0.0).margin(1e-5));
    else
    {
      REQUIRE(nbc.Variances()[i] ==
          Approx(nbcTrain.Variances()[i]).epsilon(1e-7));
    }
  }

  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    if (std::abs(nbc.Probabilities()[i]) < 1e-5)
      REQUIRE(nbcTrain.Probabilities()[i] == Approx(0.0).margin(1e-5));
    else
    {
      REQUIRE(nbc.Probabilities()[i] ==
          Approx(nbcTrain.Probabilities()[i]).epsilon(1e-7));
    }
  }
}

TEST_CASE("SeparateTrainIndividualIncrementalTest", "[NBCTest]")
{
  const char* trainFilename = "trainSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!Load(trainResultFilename, trainRes))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc(trainData, labels, classes, true);
  NaiveBayesClassifier<> nbcTrain(trainData.n_rows, classes);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    nbcTrain.Train(trainData.col(i), labels[i]);

  REQUIRE(nbc.Means().n_rows == nbcTrain.Means().n_rows);
  REQUIRE(nbc.Means().n_cols == nbcTrain.Means().n_cols);
  REQUIRE(nbc.Variances().n_rows == nbcTrain.Variances().n_rows);
  REQUIRE(nbc.Variances().n_cols == nbcTrain.Variances().n_cols);
  REQUIRE(nbc.Probabilities().n_elem ==
                      nbcTrain.Probabilities().n_elem);

  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    if (std::abs(nbc.Means()[i]) < 1e-5)
      REQUIRE(nbcTrain.Means()[i] == Approx(0.0).margin(1e-5));
    else
      REQUIRE(nbc.Means()[i] == Approx(nbcTrain.Means()[i]).epsilon(1e-7));
  }

  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    if (std::abs(nbc.Variances()[i]) < 1e-5)
      REQUIRE(nbcTrain.Variances()[i] == Approx(0.0).margin(1e-5));
    else
    {
      REQUIRE(nbc.Variances()[i] ==
          Approx(nbcTrain.Variances()[i]).epsilon(1e-7));
    }
  }

  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    if (std::abs(nbc.Probabilities()[i]) < 1e-5)
      REQUIRE(nbcTrain.Probabilities()[i] == Approx(0.0).margin(1e-5));
    else
    {
      REQUIRE(nbc.Probabilities()[i] ==
          Approx(nbcTrain.Probabilities()[i]).epsilon(1e-7));
    }
  }
}

/**
 * Check if NaiveBayesClassifier::Classify() works properly for a high
 * dimension datasets.
 */
TEST_CASE("NaiveBayesClassifierHighDimensionsTest", "[NBCTest]")
{
  // Set file names of dataset of training and test.
  // The training dataset has 5 classes and each class has 1,000 dimensions.
  const char* trainFilename = "nbc_high_dim_train.csv";
  const char* testFilename = "nbc_high_dim_test.csv";
  const char* trainLabelsFileName = "nbc_high_dim_train_labels.csv";
  const char* testLabelsFilename = "nbc_high_dim_test_labels.csv";

  size_t classes = 5;

  // Create variables for training and assign data to them.
  arma::mat trainData;
  arma::Row<size_t> trainLabels;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!Load(trainLabelsFileName, trainLabels))
    FAIL("Cannot load dataset");

  // Initialize and train a NBC model.
  NaiveBayesClassifier<> nbcTest(trainData, trainLabels, classes);

  // Create variables for test and assign data to them.
  arma::mat testData, calcProbs;
  arma::Row<size_t> testLabels;
  arma::Row<size_t> calcVec;
  if (!Load(testFilename, testData))
    FAIL("Cannot load dataset");
  if (!Load(testLabelsFilename, testLabels))
    FAIL("Cannot load dataset");

  // Classify observations in the test dataset. To use Classify() method with
  // a parameter for probabilities of predictions, we pass 'calcProbs' to the
  // method.
  nbcTest.Classify(testData, calcVec, calcProbs);

  // Check the results.
  for (size_t i = 0; i < calcVec.n_cols; ++i)
    REQUIRE(calcVec(i) == testLabels(i));
}

/**
 * Test that we can reset the model.
 */
TEST_CASE("NBCResetTest", "[NBCTest]")
{
  const char* trainFilename = "trainSet.csv";

  arma::mat trainData;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc1, nbc2;

  nbc1.Train(trainData, labels, 2);
  nbc2.Train(trainData, labels, 2);

  REQUIRE(approx_equal(nbc1.Probabilities(), nbc2.Probabilities(), "absdiff",
      1e-5));
  REQUIRE(approx_equal(nbc1.Means(), nbc2.Means(), "absdiff", 1e-5));
  REQUIRE(approx_equal(nbc1.Variances(), nbc2.Variances(), "absdiff", 1e-5));

  // Now reset one model but train the other.  Modify the training set slightly.
  trainData += 0.1 * arma::randu<arma::mat>(trainData.n_rows, trainData.n_cols);

  nbc1.Train(trainData, labels, 2);
  nbc2.Reset();
  nbc2.Train(trainData, labels, 2);

  // Both datasets have the same class frequencies, even though the number of
  // observations and the conditional distributions differ.
  REQUIRE(approx_equal(nbc1.Probabilities(), nbc2.Probabilities(), "absdiff",
      1e-5));
  REQUIRE(!approx_equal(nbc1.Means(), nbc2.Means(), "absdiff", 1e-5));
  REQUIRE(!approx_equal(nbc1.Variances(), nbc2.Variances(), "absdiff", 1e-5));
}

/**
 * Test that we can use a model for incremental point-by-point training.
 */
TEMPLATE_TEST_CASE("NBCIncrementalTest", "[NBCTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;

  const char* trainFilename = "trainSet.csv";

  MatType trainData;
  if (!Load(trainFilename, trainData))
    FAIL("Cannot load dataset");

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<MatType> nbc(trainData.n_rows, 2);

  for (size_t i = 0; i < trainData.n_cols; ++i)
  {
    nbc.Train(trainData.col(i), labels[i]);
  }

  // We don't care what the result is, we are more concerned with the fact that
  // training worked at all.
  REQUIRE(nbc.Probabilities().n_elem == 2);
  REQUIRE(nbc.Means().n_rows == trainData.n_rows);
  REQUIRE(nbc.Means().n_cols == 2);
  REQUIRE(nbc.Variances().n_rows == trainData.n_rows);
  REQUIRE(nbc.Variances().n_cols == 2);
}

/**
 * Test that we can train on sparse data with different internal model types.
 */
TEMPLATE_TEST_CASE("NBCModelMatTypeTest", "[NBCTest]", float, double)
{
  using ElemType = TestType;

  NaiveBayesClassifier<arma::Mat<ElemType>> nbc;

  // Create random data; 5000 points in 4 classes.
  arma::SpMat<ElemType> data;
  data.sprandu(100, 5000, 0.2);
  arma::Row<size_t> labels =
      arma::randi<arma::Row<size_t>>(5000, DistrParam(0, 3));

  nbc.Train(data, labels, 4);

  // We don't care what the result is, we are more concerned with the fact that
  // training worked at all.
  REQUIRE(nbc.Probabilities().n_elem == 4);
  REQUIRE(nbc.Means().n_rows == data.n_rows);
  REQUIRE(nbc.Means().n_cols == 4);
  REQUIRE(nbc.Variances().n_rows == data.n_rows);
  REQUIRE(nbc.Variances().n_cols == 4);
}

/**
 * Splitting a dataset into batches must preserve its sufficient statistics.
 */
TEMPLATE_TEST_CASE("NBCBatchPartitionTest", "[NBCTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  const MatType data = {{0, 2, 4, 10, 12, 14, 1, 3, 11, 13, 20, 22},
                       {5, 5, 5,  1,  2,  3, 5, 5,  4,  5,  6,  6}};
  const arma::Row<size_t> labels = {0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2};
  const double epsilon = 0.25;
  const NaiveBayesClassifier<MatType> expected(data, labels, 4, false, epsilon);

  for (size_t split = 1; split < data.n_cols; ++split)
  {
    for (const bool incremental : {false, true})
    {
      NaiveBayesClassifier<MatType> nbc(data.cols(0, split - 1),
          labels.cols(0, split - 1), 4, incremental, epsilon);
      nbc.Train(data.cols(split, data.n_cols - 1),
          labels.cols(split, data.n_cols - 1), 4);

      REQUIRE(nbc.TrainingPoints() == data.n_cols);
      REQUIRE(approx_equal(nbc.Probabilities(), expected.Probabilities(),
          "absdiff", 1e-6));
      REQUIRE(approx_equal(nbc.Means(), expected.Means(),
          "absdiff", 1e-5));
      REQUIRE(approx_equal(nbc.Variances(), expected.Variances(),
          "absdiff", 1e-5));

      arma::Row<size_t> predictions, expectedPredictions;
      MatType probabilities, expectedProbabilities;
      nbc.Classify(data, predictions, probabilities);
      expected.Classify(data, expectedPredictions, expectedProbabilities);
      REQUIRE(arma::all(predictions == expectedPredictions));
      REQUIRE(approx_equal(probabilities, expectedProbabilities,
          "absdiff", 1e-5));
    }
  }
}

/**
 * Batch and point updates must use the same regularized variance convention.
 */
TEMPLATE_TEST_CASE("NBCMixedIncrementalTest", "[NBCTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;
  const MatType data = {{0, 2, 4, 10, 12, 14, 1, 3, 11, 13},
                       {5, 5, 5,  1,  2,  3, 5, 5,  4,  5}};
  const arma::Row<size_t> labels = {0, 0, 0, 1, 1, 1, 0, 0, 1, 1};
  const double epsilon = 0.25;
  NaiveBayesClassifier<MatType> nbc(data.n_rows, 3, epsilon);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    if (i % 2 == 0)
      nbc.Train(data.col(i), labels[i]);
    else
      nbc.Train(data.cols(i, i), labels.cols(i, i), 3);

    const NaiveBayesClassifier<MatType> expected(data.cols(0, i),
        labels.cols(0, i), 3, false, epsilon);
    REQUIRE(nbc.TrainingPoints() == i + 1);
    REQUIRE(approx_equal(nbc.Probabilities(), expected.Probabilities(),
        "absdiff", 1e-6));
    REQUIRE(approx_equal(nbc.Means(), expected.Means(), "absdiff", 1e-5));
    REQUIRE(approx_equal(nbc.Variances(), expected.Variances(),
        "absdiff", 1e-5));
  }

  // An empty incremental batch is a no-op, including for regularization.
  const MatType variances = nbc.Variances();
  nbc.Train(MatType(data.n_rows, 0), arma::Row<size_t>(), 3);
  REQUIRE(nbc.TrainingPoints() == data.n_cols);
  REQUIRE(approx_equal(nbc.Variances(), variances, "absdiff", 0.0));
  REQUIRE(arma::accu(nbc.Probabilities()) == Approx(1.0));
}

/**
 * Non-incremental training must replace, rather than accumulate, sample counts.
 */
TEMPLATE_TEST_CASE("NBCRetrainThenIncrementTest", "[NBCTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;
  const MatType data = {{0, 2, 4, 10, 12, 14}, {5, 5, 5, 1, 2, 3}};
  const arma::Row<size_t> labels = {0, 0, 0, 1, 1, 1};
  NaiveBayesClassifier<MatType> nbc(data, labels, 2);
  nbc.Train(data.cols(0, 3), labels.cols(0, 3), 2, false);
  REQUIRE(nbc.TrainingPoints() == 4);
  nbc.Train(data.cols(4, 5), labels.cols(4, 5), 2);
  const NaiveBayesClassifier<MatType> expected(data, labels, 2);
  REQUIRE(nbc.TrainingPoints() == data.n_cols);
  REQUIRE(approx_equal(nbc.Probabilities(), expected.Probabilities(),
      "absdiff", 1e-6));
  REQUIRE(approx_equal(nbc.Means(), expected.Means(), "absdiff", 1e-5));
  REQUIRE(approx_equal(nbc.Variances(), expected.Variances(), "absdiff", 1e-5));
}

/**
 * Resizing must preserve epsilon, and incremental training must not replace it.
 */
TEMPLATE_TEST_CASE("NBCIncrementalEpsilonTest", "[NBCTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;
  const MatType data = {{2, 2, 2, 4, 4, 4}, {5, 5, 5, 6, 6, 6}};
  const arma::Row<size_t> labels = {0, 0, 0, 1, 1, 1};
  NaiveBayesClassifier<MatType> nbc(0, 0, 0.25);
  nbc.Train(data.cols(0, 2), labels.cols(0, 2), 2);
  // As documented, epsilon is ignored for incremental updates.
  nbc.Train(data.cols(3, 5), labels.cols(3, 5), 2, true, 0.5);
  REQUIRE(approx_equal(nbc.Variances(), MatType(2, 2, arma::fill::value(0.25)),
      "absdiff", 1e-6));
}

/**
 * Rebatching must not change a prediction near a class decision boundary.
 */
TEST_CASE("NBCIncrementalPredictionTest", "[NBCTest]")
{
  const arma::mat data = {{0, 2, 4, 10, 12, 14}};
  const arma::Row<size_t> labels = {0, 0, 0, 1, 1, 1};
  NaiveBayesClassifier<> nbc;
  nbc.Train(data.cols(0, 2), labels.cols(0, 2), 2);
  nbc.Train(data.cols(3, 5), labels.cols(3, 5), 2);
  size_t prediction;
  arma::vec probabilities;
  nbc.Classify(arma::vec({6.5}), prediction, probabilities);
  REQUIRE(prediction == 0);
  REQUIRE(probabilities[0] == Approx(0.7772998612));
  REQUIRE(arma::accu(probabilities) == Approx(1.0));
}
