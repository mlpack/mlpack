/**
 * @file nbc_test.cpp
 *
 * Test for the Naive Bayes classifier.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

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
  arma::Col<size_t> labels(trainData.n_cols);
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
  arma::Col<size_t> calcVec;
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
  arma::Col<size_t> labels(trainData.n_cols);
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
  arma::Col<size_t> calcVec;
  data::Load(testFilename, testData, true);
  data::Load(testResultFilename, testRes, true);

  testData.shed_row(testData.n_rows - 1); // Remove the labels.

  nbcTest.Classify(testData, calcVec);

  for (size_t i = 0; i < testData.n_cols; i++)
    BOOST_REQUIRE_EQUAL(testRes(i), calcVec(i));
}

BOOST_AUTO_TEST_SUITE_END();
