/**
 * @file perceptron_test.cpp
 * @author Udit Saxena
 *
 * Tests for perceptron.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::perceptron;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(PerceptronTest);

/**
 * This test tests whether the perceptron converges for the AND gate classifier.
 */
BOOST_AUTO_TEST_CASE(And)
{
  mat trainData;
  trainData << 0 << 1 << 1 << 0 << endr
            << 1 << 0 << 1 << 0 << endr;
  Mat<size_t> labels;
  labels << 0 << 0 << 1 << 0;

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData << 0 << 1 << 1 << 0 << endr
           << 1 << 0 << 1 << 0 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0, 0), 0);
  BOOST_CHECK_EQUAL(predictedLabels(0, 1), 0);
  BOOST_CHECK_EQUAL(predictedLabels(0, 2), 1);
  BOOST_CHECK_EQUAL(predictedLabels(0, 3), 0);
}

/**
 * This test tests whether the perceptron converges for the OR gate classifier.
 */
BOOST_AUTO_TEST_CASE(Or)
{
  mat trainData;
  trainData << 0 << 1 << 1 << 0 << endr
            << 1 << 0 << 1 << 0 << endr;

  Mat<size_t> labels;
  labels << 1 << 1 << 1 << 0;

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData << 0 << 1 << 1 << 0 << endr
           << 1 << 0 << 1 << 0 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0, 0), 1);
  BOOST_CHECK_EQUAL(predictedLabels(0, 1), 1);
  BOOST_CHECK_EQUAL(predictedLabels(0, 2), 1);
  BOOST_CHECK_EQUAL(predictedLabels(0, 3), 0);
}

/**
 * This tests the convergence on a set of linearly separable data with 3
 * classes.
 */
BOOST_AUTO_TEST_CASE(Random3)
{
  mat trainData;
  trainData << 0 << 1 << 1 << 4 << 5 << 4 << 1 << 2 << 1 << endr
            << 1 << 0 << 1 << 1 << 1 << 2 << 4 << 5 << 4 << endr;

  Mat<size_t> labels;
  labels << 0 << 0 << 0 << 1 << 1 << 1 << 2 << 2 << 2;

  Perceptron<> p(trainData, labels.row(0), 3, 1000);

  mat testData;
  testData << 0 << 1 << 1 << endr
           << 1 << 0 << 1 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);

  for (size_t i = 0; i < predictedLabels.n_cols; i++)
    BOOST_CHECK_EQUAL(predictedLabels(0, i), 0);

}

/**
 * This tests the convergence of the perceptron on a dataset which has only TWO
 * points which belong to different classes.
 */
BOOST_AUTO_TEST_CASE(TwoPoints)
{
  mat trainData;
  trainData << 0 << 1 << endr
            << 1 << 0 << endr;

  Mat<size_t> labels;
  labels << 0 << 1;

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData << 0 << 1 << endr
           << 1 << 0 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0, 0), 0);
  BOOST_CHECK_EQUAL(predictedLabels(0, 1), 1);
}

/**
 * This tests the convergence of the perceptron on a dataset which has a
 * non-linearly separable dataset.
 */
BOOST_AUTO_TEST_CASE(NonLinearlySeparableDataset)
{
  mat trainData;
  trainData << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8
            << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << endr
            << 1 << 1 << 1 << 1 << 1 << 1 << 1 << 1
            << 2 << 2 << 2 << 2 << 2 << 2 << 2 << 2 << endr;

  Mat<size_t> labels;
  labels << 0 << 0 << 0 << 1 << 0 << 1 << 1 << 1
         << 0 << 0 << 0 << 1 << 0 << 1 << 1 << 1;

  Perceptron<> p(trainData, labels.row(0), 2, 1000);

  mat testData;
  testData << 3 << 4   << 5   << 6   << endr
           << 3 << 2.3 << 1.7 << 1.5 << endr;
  Row<size_t> predictedLabels(testData.n_cols);
  p.Classify(testData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0, 0), 0);
  BOOST_CHECK_EQUAL(predictedLabels(0, 1), 0);
  BOOST_CHECK_EQUAL(predictedLabels(0, 2), 1);
  BOOST_CHECK_EQUAL(predictedLabels(0, 3), 1);
}

BOOST_AUTO_TEST_CASE(SecondaryConstructor)
{
  mat trainData;
  trainData << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8
            << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << endr
            << 1 << 1 << 1 << 1 << 1 << 1 << 1 << 1
            << 2 << 2 << 2 << 2 << 2 << 2 << 2 << 2 << endr;

  Mat<size_t> labels;
  labels << 0 << 0 << 0 << 1 << 0 << 1 << 1 << 1
         << 0 << 0 << 0 << 1 << 0 << 1 << 1 << 1;

  Perceptron<> p1(trainData, labels.row(0), 2, 1000);

  Perceptron<> p2(p1);
}

BOOST_AUTO_TEST_SUITE_END();
