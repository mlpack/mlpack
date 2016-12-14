/**
 * @file nbc_test.cpp
 *
 * Test for the Naive Bayes classifier.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace naive_bayes;

BOOST_AUTO_TEST_SUITE(NBCTest);

BOOST_AUTO_TEST_CASE(NaiveBayesClassifierTest)
{
  const char* trainFilename = "trainSet.csv";
  const char* testFilename = "testSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  const char* testResultFilename = "testRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  data::Load(trainFilename, trainData, true);
  data::Load(trainResultFilename, trainRes, true);

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbcTest(trainData, labels, classes);

  size_t dimension = nbcTest.Means().n_rows;
  calcMat.zeros(2 * dimension + 1, classes);

  for (size_t i = 0; i < dimension; i++)
  {
    for (size_t j = 0; j < classes; j++)
    {
      calcMat(i, j) = nbcTest.Means()(i, j);
      calcMat(i + dimension, j) = nbcTest.Variances()(i, j);
    }
  }

  for (size_t i = 0; i < classes; i++)
    calcMat(2 * dimension, i) = nbcTest.Probabilities()(i);

  for (size_t i = 0; i < calcMat.n_rows; i++)
    for (size_t j = 0; j < classes; j++)
      BOOST_REQUIRE_CLOSE(trainRes(i, j) + .00001, calcMat(i, j), 0.01);

  arma::mat testData;
  arma::Mat<size_t> testRes;
  arma::Row<size_t> calcVec;
  data::Load(testFilename, testData, true);
  data::Load(testResultFilename, testRes, true);

  testData.shed_row(testData.n_rows - 1); // Remove the labels.

  nbcTest.Classify(testData, calcVec);

  for (size_t i = 0; i < testData.n_cols; i++)
    BOOST_REQUIRE_EQUAL(testRes(i), calcVec(i));
}

// The same test, but this one uses the incremental algorithm to calculate
// variance.
BOOST_AUTO_TEST_CASE(NaiveBayesClassifierIncrementalTest)
{
  const char* trainFilename = "trainSet.csv";
  const char* testFilename = "testSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  const char* testResultFilename = "testRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  data::Load(trainFilename, trainData, true);
  data::Load(trainResultFilename, trainRes, true);

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbcTest(trainData, labels, classes, true);

  size_t dimension = nbcTest.Means().n_rows;
  calcMat.zeros(2 * dimension + 1, classes);

  for (size_t i = 0; i < dimension; i++)
  {
    for (size_t j = 0; j < classes; j++)
    {
      calcMat(i, j) = nbcTest.Means()(i, j);
      calcMat(i + dimension, j) = nbcTest.Variances()(i, j);
    }
  }

  for (size_t i = 0; i < classes; i++)
    calcMat(2 * dimension, i) = nbcTest.Probabilities()(i);

  for (size_t i = 0; i < calcMat.n_rows; i++)
    for (size_t j = 0; j < classes; j++)
      BOOST_REQUIRE_CLOSE(trainRes(i, j) + .00001, calcMat(i, j), 0.01);

  arma::mat testData;
  arma::Mat<size_t> testRes;
  arma::Row<size_t> calcVec;
  data::Load(testFilename, testData, true);
  data::Load(testResultFilename, testRes, true);

  testData.shed_row(testData.n_rows - 1); // Remove the labels.

  nbcTest.Classify(testData, calcVec);

  for (size_t i = 0; i < testData.n_cols; i++)
    BOOST_REQUIRE_EQUAL(testRes(i), calcVec(i));
}

/**
 * Ensure that separate training gives the same model.
 */
BOOST_AUTO_TEST_CASE(SeparateTrainTest)
{
  const char* trainFilename = "trainSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  data::Load(trainFilename, trainData, true);
  data::Load(trainResultFilename, trainRes, true);

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc(trainData, labels, classes, true);
  NaiveBayesClassifier<> nbcTrain(trainData.n_rows, classes);
  nbcTrain.Train(trainData, labels, false);

  BOOST_REQUIRE_EQUAL(nbc.Means().n_rows, nbcTrain.Means().n_rows);
  BOOST_REQUIRE_EQUAL(nbc.Means().n_cols, nbcTrain.Means().n_cols);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_rows, nbcTrain.Variances().n_rows);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_cols, nbcTrain.Variances().n_cols);
  BOOST_REQUIRE_EQUAL(nbc.Probabilities().n_elem,
                      nbcTrain.Probabilities().n_elem);

  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    if (std::abs(nbc.Means()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Means()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Means()[i], nbcTrain.Means()[i], 1e-5);
  }

  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    if (std::abs(nbc.Variances()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Variances()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Variances()[i], nbcTrain.Variances()[i], 1e-5);
  }

  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    if (std::abs(nbc.Probabilities()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Probabilities()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Probabilities()[i], nbcTrain.Probabilities()[i],
          1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SeparateTrainIncrementalTest)
{
  const char* trainFilename = "trainSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  data::Load(trainFilename, trainData, true);
  data::Load(trainResultFilename, trainRes, true);

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc(trainData, labels, classes, true);
  NaiveBayesClassifier<> nbcTrain(trainData.n_rows, classes);
  nbcTrain.Train(trainData, labels, true);

  BOOST_REQUIRE_EQUAL(nbc.Means().n_rows, nbcTrain.Means().n_rows);
  BOOST_REQUIRE_EQUAL(nbc.Means().n_cols, nbcTrain.Means().n_cols);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_rows, nbcTrain.Variances().n_rows);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_cols, nbcTrain.Variances().n_cols);
  BOOST_REQUIRE_EQUAL(nbc.Probabilities().n_elem,
                      nbcTrain.Probabilities().n_elem);

  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    if (std::abs(nbc.Means()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Means()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Means()[i], nbcTrain.Means()[i], 1e-5);
  }

  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    if (std::abs(nbc.Variances()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Variances()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Variances()[i], nbcTrain.Variances()[i], 1e-5);
  }

  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    if (std::abs(nbc.Probabilities()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Probabilities()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Probabilities()[i], nbcTrain.Probabilities()[i],
          1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SeparateTrainIndividualIncrementalTest)
{
  const char* trainFilename = "trainSet.csv";
  const char* trainResultFilename = "trainRes.csv";
  size_t classes = 2;

  arma::mat trainData, trainRes, calcMat;
  data::Load(trainFilename, trainData, true);
  data::Load(trainResultFilename, trainRes, true);

  // Get the labels out.
  arma::Row<size_t> labels(trainData.n_cols);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    labels[i] = trainData(trainData.n_rows - 1, i);
  trainData.shed_row(trainData.n_rows - 1);

  NaiveBayesClassifier<> nbc(trainData, labels, classes, true);
  NaiveBayesClassifier<> nbcTrain(trainData.n_rows, classes);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    nbcTrain.Train(trainData.col(i), labels[i]);

  BOOST_REQUIRE_EQUAL(nbc.Means().n_rows, nbcTrain.Means().n_rows);
  BOOST_REQUIRE_EQUAL(nbc.Means().n_cols, nbcTrain.Means().n_cols);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_rows, nbcTrain.Variances().n_rows);
  BOOST_REQUIRE_EQUAL(nbc.Variances().n_cols, nbcTrain.Variances().n_cols);
  BOOST_REQUIRE_EQUAL(nbc.Probabilities().n_elem,
                      nbcTrain.Probabilities().n_elem);

  for (size_t i = 0; i < nbc.Means().n_elem; ++i)
  {
    if (std::abs(nbc.Means()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Means()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Means()[i], nbcTrain.Means()[i], 1e-5);
  }

  for (size_t i = 0; i < nbc.Variances().n_elem; ++i)
  {
    if (std::abs(nbc.Variances()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Variances()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Variances()[i], nbcTrain.Variances()[i], 1e-5);
  }

  for (size_t i = 0; i < nbc.Probabilities().n_elem; ++i)
  {
    if (std::abs(nbc.Probabilities()[i]) < 1e-5)
      BOOST_REQUIRE_SMALL(nbcTrain.Probabilities()[i], 1e-5);
    else
      BOOST_REQUIRE_CLOSE(nbc.Probabilities()[i], nbcTrain.Probabilities()[i],
          1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
