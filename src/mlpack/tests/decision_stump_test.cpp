/**
 * @file decision_stump_test.cpp
 * @author Udit Saxena
 *
 * Tests for DecisionStump class.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_stump/decision_stump.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::decision_stump;
using namespace arma;

BOOST_AUTO_TEST_SUITE(DecisionStumpTest);

/**
 * This tests handles the case wherein only one class exists in the input
 * labels.  It checks whether the only class supplied was the only class
 * predicted.
 */
BOOST_AUTO_TEST_CASE(OneClass)
{
  const size_t numClasses = 2;
  const size_t inpBucketSize = 6;

  mat trainingData;
  trainingData << 2.4 << 3.8 << 3.8 << endr
               << 1   << 1   << 2   << endr
               << 1.3 << 1.9 << 1.3 << endr;

  // No need to normalize labels here.
  Mat<size_t> labelsIn;
  labelsIn << 1 << 1 << 1;

  mat testingData;
  testingData << 2.4 << 2.5 << 2.6;

  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  for (size_t i = 0; i < predictedLabels.size(); i++ )
    BOOST_CHECK_EQUAL(predictedLabels(i), 1);

}

/**
 * This tests for the classification:
 *   if testinput < 0 - class 0
 *   if testinput > 0 - class 1
 * An almost perfect split on zero.
 */
BOOST_AUTO_TEST_CASE(PerfectSplitOnZero)
{
  const size_t numClasses = 2;
  const size_t inpBucketSize = 2;

  mat trainingData;
  trainingData << -1 << 1 << -2 << 2 << -3 << 3;

  // No need to normalize labels here.
  Mat<size_t> labelsIn;
  labelsIn << 0 << 1 << 0 << 1 << 0 << 1;

  mat testingData;
  testingData << -4 << 7 << -7 << -5 << 6;

  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0,0),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,1),1);
  BOOST_CHECK_EQUAL(predictedLabels(0,2),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,3),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,4),1);
}

/**
 * This tests the binning function for the case when a dataset with cardinality
 * of input < inpBucketSize is provided.
 */
BOOST_AUTO_TEST_CASE(BinningTesting)
{
  const size_t numClasses = 2;
  const size_t inpBucketSize = 10;

  mat trainingData;
  trainingData << -1 << 1 << -2 << 2 << -3 << 3 << -4;

  // No need to normalize labels here.
  Mat<size_t> labelsIn;
  labelsIn << 0 << 1 << 0 << 1 << 0 << 1 << 0;

  mat testingData;
  testingData << 5;

  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0,0),0);
}

/**
 * This is a test for the case when non-overlapping, multiple classes are
 * provided. It tests for a perfect split due to the non-overlapping nature of
 * the input classes.
 */
BOOST_AUTO_TEST_CASE(PerfectMultiClassSplit)
{
  const size_t numClasses = 4;
  const size_t inpBucketSize = 3;

  mat trainingData;
  trainingData << -8 << -7 << -6 << -5 << -4 << -3 << -2 << -1
               << 0  << 1  << 2  << 3  << 4  << 5  << 6  << 7;

  // No need to normalize labels here.
  Mat<size_t> labelsIn;
  labelsIn << 0 << 0 << 0 << 0 << 1 << 1 << 1 << 1
           << 2 << 2 << 2 << 2 << 3 << 3 << 3 << 3;

  mat testingData;
  testingData << -6.1 << -2.1 << 1.1 << 5.1;

  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0,0),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,1),1);
  BOOST_CHECK_EQUAL(predictedLabels(0,2),2);
  BOOST_CHECK_EQUAL(predictedLabels(0,3),3);
}

/**
 * This test is for the case when reasonably overlapping, multiple classes are
 * provided in the input label set. It tests whether classification takes place
 * with a reasonable amount of error due to the overlapping nature of input
 * classes.
 */
BOOST_AUTO_TEST_CASE(MultiClassSplit)
{
  const size_t numClasses = 3;
  const size_t inpBucketSize = 3;

  mat trainingData;
  trainingData << -7 << -6 << -5 << -4 << -3 << -2 << -1 << 0 << 1
               << 2  << 3  << 4  << 5  << 6  << 7  << 8  << 9 << 10;

  // No need to normalize labels here.
  Mat<size_t> labelsIn;
  labelsIn << 0 << 0 << 0 << 0 << 1 << 1 << 0 << 0
           << 1 << 1 << 1 << 2 << 1 << 2 << 2 << 2 << 2 << 2;


  mat testingData;
  testingData << -6.1 << -5.9 << -2.1 << -0.7 << 2.5 << 4.7 << 7.2 << 9.1;

  DecisionStump<> ds(trainingData, labelsIn.row(0), numClasses, inpBucketSize);

  Row<size_t> predictedLabels(testingData.n_cols);
  ds.Classify(testingData, predictedLabels);

  BOOST_CHECK_EQUAL(predictedLabels(0,0),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,1),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,2),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,3),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,4),0);
  BOOST_CHECK_EQUAL(predictedLabels(0,5),1);
  BOOST_CHECK_EQUAL(predictedLabels(0,6),2);
  BOOST_CHECK_EQUAL(predictedLabels(0,7),2);
}

BOOST_AUTO_TEST_SUITE_END();
