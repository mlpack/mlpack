/**
 * @file Adaboost_test.cpp
 * @author Udit Saxena
 *
 * Tests for Adaboost class.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/adaboost/adaboost.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::adaboost;

BOOST_AUTO_TEST_SUITE(AdaboostTest);

/**
 *  This test case runs the Adaboost.mh algorithm on the UCI Iris dataset.
 *  It checks whether the hamming loss breaches the upperbound, which
 *  is provided by ztAccumulator.
 */
BOOST_AUTO_TEST_CASE(HammingLossBound)
{
  arma::mat inputData;

  if (!data::Load("iris.txt", inputData))
    BOOST_FAIL("Cannot load test dataset iris.txt!");

  arma::Mat<size_t> labels;

  if (!data::Load("iris_labels.txt",labels))
    BOOST_FAIL("Cannot load labels for iris iris_labels.txt");
  
  // no need to map the labels here

  // Define your own weak learner, perceptron in this case.
  // Run the perceptron for perceptron_iter iterations.
  int perceptron_iter = 4000;

  perceptron::Perceptron<> p(inputData, labels.row(0), perceptron_iter);

  // Define parameters for the adaboost
  int iterations = 100;
  Adaboost<> a(inputData, labels.row(0), iterations, p);
  int countError = 0;
  for (size_t i = 0; i < labels.n_cols; i++)
    if(labels(i) != a.finalHypothesis(i))
      countError++;
  double hammingLoss = (double) countError / labels.n_cols;

  BOOST_REQUIRE(hammingLoss <= a.ztAccumulator);
}

BOOST_AUTO_TEST_SUITE_END();