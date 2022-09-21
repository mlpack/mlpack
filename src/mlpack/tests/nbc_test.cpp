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

TEST_CASE("NaiveBayesClassifierTest", "[NBCTest]")
{
  const char* trainFilename = "trainSet.csv";
  const char* testFilename = "testSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  const char* testResultFilename = "testRes.csv";
  const char* testResultProbsFilename = "testResProbs.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  if (!data::Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!data::Load(trainResultFilename, trainRes))
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
  if (!data::Load(testFilename, testData))
    FAIL("Cannot load dataset");
  if (!data::Load(testResultFilename, testRes))
    FAIL("Cannot load dataset");
  if (!data::Load(testResultProbsFilename, testResProbs))
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
  if (!data::Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!data::Load(trainResultFilename, trainRes))
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
  if (!data::Load(testFilename, testData))
    FAIL("Cannot load dataset");
  if (!data::Load(testResultFilename, testRes))
    FAIL("Cannot load dataset");
  if (!data::Load(testResultProbsFilename, testResProba))
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
  if (!data::Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!data::Load(trainResultFilename, trainRes))
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
  if (!data::Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!data::Load(trainResultFilename, trainRes))
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
  if (!data::Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!data::Load(trainResultFilename, trainRes))
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
  if (!data::Load(trainFilename, trainData))
    FAIL("Cannot load dataset");
  if (!data::Load(trainLabelsFileName, trainLabels))
    FAIL("Cannot load dataset");

  // Initialize and train a NBC model.
  NaiveBayesClassifier<> nbcTest(trainData, trainLabels, classes);

  // Create variables for test and assign data to them.
  arma::mat testData, calcProbs;
  arma::Row<size_t> testLabels;
  arma::Row<size_t> calcVec;
  if (!data::Load(testFilename, testData))
    FAIL("Cannot load dataset");
  if (!data::Load(testLabelsFilename, testLabels))
    FAIL("Cannot load dataset");

  // Classify observations in the test dataset. To use Classify() method with
  // a parameter for probabilities of predictions, we pass 'calcProbs' to the
  // method.
  nbcTest.Classify(testData, calcVec, calcProbs);

  // Check the results.
  for (size_t i = 0; i < calcVec.n_cols; ++i)
    REQUIRE(calcVec(i) == testLabels(i));
}
