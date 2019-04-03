/**
 * @file cv_test.cpp
 *
 * Unit tests for the cross-validation module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <type_traits>

#include <mlpack/core/cv/meta_info_extractor.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/metrics/f1.hpp>
#include <mlpack/core/cv/metrics/mse.hpp>
#include <mlpack/core/cv/metrics/precision.hpp>
#include <mlpack/core/cv/metrics/recall.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/cv/k_fold_cv.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>
#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "mock_categorical_data.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::cv;
using namespace mlpack::naive_bayes;
using namespace mlpack::regression;
using namespace mlpack::tree;
using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(CVTest);

/**
 * Test metrics for binary classification.
 */
BOOST_AUTO_TEST_CASE(BinaryClassificationMetricsTest)
{
  // Using the same data for training and testing.
  arma::mat data = arma::linspace<arma::rowvec>(1.0, 10.0, 10);

  // Labels that will be considered as "ground truth".
  arma::Row<size_t> labels("0 0 1 0 0  1 0 1 0 1");

  // Labels that make the data linearly separable. These labels will be
  // predicted in response to the data since we use them for training.
  arma::Row<size_t> predictedLabels("0 0 0 0 0  1 1 1 1 1");

  LogisticRegression<> lr(data, predictedLabels);

  BOOST_REQUIRE_CLOSE(Accuracy::Evaluate(lr, data, labels), 0.7, 1e-5);

  BOOST_REQUIRE_CLOSE(Precision<Binary>::Evaluate(lr, data, labels), 0.6, 1e-5);

  BOOST_REQUIRE_CLOSE(Recall<Binary>::Evaluate(lr, data, labels), 0.75, 1e-5);

  double f1 = 2 * 0.6 * 0.75 / (0.6 + 0.75);
  BOOST_REQUIRE_CLOSE(F1<Binary>::Evaluate(lr, data, labels), f1, 1e-5);
}

/**
 * Test for confusion matrix.
 */
BOOST_AUTO_TEST_CASE(ConfusionMatrixTest)
{
  // Labels that will be considered as "ground truth".
  arma::Row<size_t> labels("0 0 1 0 0  1 0 1 0 1");

  // Predicted labels.
  arma::Row<size_t> predictedLabels("0 0 0 0 0  1 1 1 1 1");
  // Confusion matrix.
  arma::Mat<int> output;
  data::ConfusionMatrix(predictedLabels, labels, output, 2);
  BOOST_REQUIRE_EQUAL(output(0, 0), 4);
  BOOST_REQUIRE_EQUAL(output(0, 1), 1);
  BOOST_REQUIRE_EQUAL(output(1, 0), 2);
  BOOST_REQUIRE_EQUAL(output(1, 1), 3);
}

/**
 * Test metrics for multiclass classification.
 */
BOOST_AUTO_TEST_CASE(MulticlassClassificationMetricsTest)
{
  // Using the same data for training and testing.
  arma::mat data = arma::linspace<arma::rowvec>(1.0, 12.0, 12);

  // Labels that will be considered as "ground truth".
  arma::Row<size_t> labels("0 1  0 1  2 2 1 2  3 3 3 3");

  // These labels should be predicted in response to the data since we use them
  // for training.
  arma::Row<size_t> predictedLabels("0 0  1 1  2 2 2 2  3 3 3 3");
  size_t numClasses = 4;

  NaiveBayesClassifier<> nb(data, predictedLabels, numClasses);

  // Assert that the Naive Bayes model really predicts the labels above in
  // response to the data.
  BOOST_REQUIRE_CLOSE(Accuracy::Evaluate(nb, data, predictedLabels), 1.0, 1e-5);

  double microaveragedPrecision = double(1 + 1 + 3 + 4) / 12;
  BOOST_REQUIRE_CLOSE(Precision<Micro>::Evaluate(nb, data, labels),
      microaveragedPrecision, 1e-5);

  double microaveragedRecall = double(1 + 1 + 3 + 4) / 12;
  BOOST_REQUIRE_CLOSE(Recall<Micro>::Evaluate(nb, data, labels),
      microaveragedRecall, 1e-5);

  double microaveragedF1 = 2 * microaveragedPrecision * microaveragedRecall /
    (microaveragedPrecision + microaveragedRecall);
  BOOST_REQUIRE_CLOSE(F1<Micro>::Evaluate(nb, data, labels),
      microaveragedF1, 1e-5);

  double macroaveragedPrecision = (0.5 + 0.5 + 0.75 + 1.0) / 4;
  BOOST_REQUIRE_CLOSE(Precision<Macro>::Evaluate(nb, data, labels),
      macroaveragedPrecision, 1e-5);

  double macroaveragedRecall = (0.5 + 1.0 / 3 + 1.0 + 1.0) / 4;
  BOOST_REQUIRE_CLOSE(Recall<Macro>::Evaluate(nb, data, labels),
      macroaveragedRecall, 1e-5);

  double macroaveragedF1 = (2 * 0.5 * 0.5 / (0.5 + 0.5) +
      2 * 0.5 * (1.0 / 3) / (0.5 + (1.0 / 3)) + 2 * 0.75 * 1.0 / (0.75 + 1.0) +
      2 * 1.0 * 1.0 / (1.0 + 1.0)) / 4;
  BOOST_REQUIRE_CLOSE(F1<Macro>::Evaluate(nb, data, labels),
      macroaveragedF1, 1e-5);
}

/**
 * Test the mean squared error.
 */
BOOST_AUTO_TEST_CASE(MSETest)
{
  // Making two points that define the linear function f(x) = x - 1
  arma::mat trainingData("0 1");
  arma::rowvec trainingResponses("-1 0");

  LinearRegression lr(trainingData, trainingResponses);

  // Making three responses that differ from the correct ones by 0, 1, and 2
  // respectively
  arma::mat data("2 3 4");
  arma::rowvec responses("1 3 5");

  double expectedMSE = (0 * 0 + 1 * 1 + 2 * 2) / 3.0;

  BOOST_REQUIRE_CLOSE(MSE::Evaluate(lr, data, responses), expectedMSE, 1e-5);
}

/**
 * Test the mean squared error with matrix responses.
 */
BOOST_AUTO_TEST_CASE(MSEMatResponsesTest)
{
  arma::mat data("1 2");
  arma::mat trainingResponses("1 2; 3 4");

  FFN<MeanSquaredError<>, ConstInitialization> ffn(MeanSquaredError<>(),
    ConstInitialization(0));
  ffn.Add<Linear<>>(1, 2);
  ffn.Add<IdentityLayer<>>();

  ens::RMSProp opt(0.2);
  opt.BatchSize() = 1;
  opt.Shuffle() = false;
  ffn.Train(data, trainingResponses, opt);

  // Making four responses that differ from the correct ones by 0, 1, 2 and 3
  // respectively
  arma::mat responses("1 3; 5 7");

  double expectedMSE = (0 * 0 + 1 * 1 + 2 * 2 + 3 * 3) / 4.0;

  BOOST_REQUIRE_CLOSE(MSE::Evaluate(ffn, data, responses), expectedMSE, 1e-1);
}

template<typename Class,
         typename ExpectedPT,
         typename PassedMT = arma::mat,
         typename PassedPT = arma::Row<size_t>>
void CheckPredictionsType()
{
  using Extractor = MetaInfoExtractor<Class, PassedMT, PassedPT>;
  using ActualPT = typename Extractor::PredictionsType;
  static_assert(std::is_same<ExpectedPT, ActualPT>::value,
      "Should be the same");
}

/**
 * Test MetaInfoExtractor correctly recognizes the type of predictions for a
 * given machine learning algorithm.
 */
BOOST_AUTO_TEST_CASE(PredictionsTypeTest)
{
  CheckPredictionsType<LinearRegression, arma::rowvec>();
  // CheckPredictionsType<FFN<>, arma::mat>();

  CheckPredictionsType<LogisticRegression<>, arma::Row<size_t>>();
  CheckPredictionsType<SoftmaxRegression, arma::Row<size_t>>();
  CheckPredictionsType<HoeffdingTree<>, arma::Row<size_t>, arma::mat>();
  CheckPredictionsType<HoeffdingTree<>, arma::Row<size_t>, arma::imat>();
  CheckPredictionsType<DecisionTree<>, arma::Row<size_t>, arma::mat,
      arma::Row<size_t>>();
  CheckPredictionsType<DecisionTree<>, arma::Row<char>, arma::mat,
      arma::Row<char>>();
}

/**
 * Test MetaInfoExtractor correctly identifies whether a given machine learning
 * algorithm supports weighted learning.
 */
BOOST_AUTO_TEST_CASE(SupportsWeightsTest)
{
  static_assert(MetaInfoExtractor<LinearRegression>::SupportsWeights,
      "Value should be true");
  static_assert(MetaInfoExtractor<DecisionTree<>>::SupportsWeights,
      "Value should be true");
  static_assert(MetaInfoExtractor<DecisionTree<>, arma::mat, arma::urowvec,
      arma::Row<float>>::SupportsWeights, "Value should be true");

  static_assert(!MetaInfoExtractor<LARS>::SupportsWeights,
      "Value should be false");
  static_assert(!MetaInfoExtractor<LogisticRegression<>>::SupportsWeights,
      "Value should be false");
}

template<typename Class,
         typename ExpectedWT,
         typename PassedMT = arma::mat,
         typename PassedPT = arma::Row<size_t>,
         typename PassedWT = arma::rowvec>
void CheckWeightsType()
{
  using Extractor = MetaInfoExtractor<Class, PassedMT, PassedPT, PassedWT>;
  using ActualWT = typename Extractor::WeightsType;
  static_assert(std::is_same<ExpectedWT, ActualWT>::value,
      "Should be the same");
}

/**
 * Test MetaInfoExtractor correctly recognizes the type of weights for a given
 * machine learning algorithm.
 */
BOOST_AUTO_TEST_CASE(WeightsTypeTest)
{
  CheckWeightsType<LinearRegression, arma::rowvec>();
  CheckWeightsType<DecisionTree<>, arma::rowvec>();
  CheckWeightsType<DecisionTree<>, arma::Row<float>, arma::mat,
      arma::Row<size_t>, arma::Row<float>>();
}

/**
 * Test MetaInfoExtractor correctly identifies whether a given machine learning
 * algorithm takes a data:DatasetInfo parameter.
 */
BOOST_AUTO_TEST_CASE(TakesDatasetInfoTest)
{
  static_assert(MetaInfoExtractor<DecisionTree<>>::TakesDatasetInfo,
      "Value should be true");
  static_assert(!MetaInfoExtractor<LinearRegression>::TakesDatasetInfo,
      "Value should be false");
  static_assert(!MetaInfoExtractor<SoftmaxRegression>::TakesDatasetInfo,
      "Value should be false");
}

/**
 * Test MetaInfoExtractor correctly identifies whether a given machine learning
 * algorithm takes the numClasses parameter.
 */
BOOST_AUTO_TEST_CASE(TakesNumClassesTest)
{
  static_assert(MetaInfoExtractor<DecisionTree<>>::TakesNumClasses,
      "Value should be true");
  static_assert(MetaInfoExtractor<SoftmaxRegression>::TakesNumClasses,
      "Value should be true");
  static_assert(!MetaInfoExtractor<LinearRegression>::TakesNumClasses,
      "Value should be false");
  static_assert(!MetaInfoExtractor<LARS>::TakesNumClasses,
      "Value should be false");
}

/**
 * Test the simple cross-validation strategy implementation with the Accuracy
 * metric.
 */
BOOST_AUTO_TEST_CASE(SimpleCVAccuracyTest)
{
  // Using the first half of data for training and the rest for validation.
  // The validation labels are 75% correct.
  arma::mat data =
    arma::mat("1 0; 2 0; 1 1; 2 1; 1 0; 2 0; 1 1; 2 1").t();
  arma::Row<size_t> labels("0 0 1 1 0 1 1 1");

  SimpleCV<LogisticRegression<>, Accuracy> cv(0.5, data, labels);

  BOOST_REQUIRE_CLOSE(cv.Evaluate(), 0.75, 1e-5);
}

/**
 * Test the simple cross-validation strategy implementation with the MSE metric.
 */
BOOST_AUTO_TEST_CASE(SimpleCVMSETest)
{
  // Using the first two points for training and remaining three for validation.
  // See the test MSETest for more explanation.
  arma::mat data("0 1 2 3 4");
  arma::rowvec responses("-1 0 1 3 5");

  double expectedMSE = (0 * 0 + 1 * 1 + 2 * 2) / 3.0;

  SimpleCV<LinearRegression, MSE> cv(0.6, data, responses);

  BOOST_REQUIRE_CLOSE(cv.Evaluate(), expectedMSE, 1e-5);

  arma::mat noiseData("-1 -2 -3 -4 -5");
  arma::rowvec noiseResponses("10 20 30 40 50");

  arma::mat allData = arma::join_rows(noiseData, data);
  arma::rowvec allResponces = arma::join_rows(noiseResponses, responses);

  arma::rowvec weights = arma::join_rows(arma::zeros(noiseData.n_cols).t(),
      arma::ones(data.n_cols).t());

  SimpleCV<LinearRegression, MSE> weightedCV(0.3, allData, allResponces,
      weights);

  BOOST_REQUIRE_CLOSE(weightedCV.Evaluate(), expectedMSE, 1e-5);

  arma::rowvec weights2 = arma::join_rows(arma::zeros(noiseData.n_cols - 1).t(),
      arma::ones(data.n_cols + 1).t());

  SimpleCV<LinearRegression, MSE> weightedCV2(0.3, allData, allResponces,
      weights2);

  BOOST_REQUIRE_GT(std::abs(weightedCV2.Evaluate() - expectedMSE), 1e-5);
}

template<typename... DTArgs>
arma::Row<size_t> PredictLabelsWithDT(const arma::mat& data,
                                      const DTArgs&... args)
{
  DecisionTree<InformationGain> dt(args...);
  arma::Row<size_t> predictedLabels;
  dt.Classify(data, predictedLabels);
  return predictedLabels;
}

/**
 * Test the simple cross-validation strategy implementation with decision trees
 * constructed in multiple ways.
 */
BOOST_AUTO_TEST_CASE(SimpleCVWithDTTest)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  arma::mat trainingData = data.cols(0, 1999);
  arma::mat testData = data.cols(2000, 3999);
  arma::Row<size_t> trainingLabels = labels.subvec(0, 1999);

  arma::rowvec weights(4000, arma::fill::randu);

  size_t numClasses = 5;
  size_t minimumLeafSize = 8;

  {
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(testData,
        trainingData, trainingLabels, numClasses, minimumLeafSize);
    SimpleCV<DecisionTree<InformationGain>, Accuracy> cv(0.5, data,
        arma::join_rows(trainingLabels, predictedLabels), numClasses);
    BOOST_REQUIRE_CLOSE(cv.Evaluate(minimumLeafSize), 1.0, 1e-5);
  }
  {
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(testData,
        trainingData, datasetInfo, trainingLabels, numClasses, minimumLeafSize);
    SimpleCV<DecisionTree<InformationGain>, Accuracy> cv(0.5, data, datasetInfo,
        arma::join_rows(trainingLabels, predictedLabels), numClasses);
    BOOST_REQUIRE_CLOSE(cv.Evaluate(minimumLeafSize), 1.0, 1e-5);
  }
  {
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(testData,
        trainingData, trainingLabels, numClasses, weights, minimumLeafSize);
    SimpleCV<DecisionTree<InformationGain>, Accuracy> cv(0.5, data,
        arma::join_rows(trainingLabels, predictedLabels), numClasses, weights);
    BOOST_REQUIRE_CLOSE(cv.Evaluate(minimumLeafSize), 1.0, 1e-5);
  }
  {
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(testData,
        trainingData, datasetInfo, trainingLabels, numClasses, weights,
        minimumLeafSize);
    SimpleCV<DecisionTree<InformationGain>, Accuracy> cv(0.5, data, datasetInfo,
        arma::join_rows(trainingLabels, predictedLabels), numClasses, weights);
    BOOST_REQUIRE_CLOSE(cv.Evaluate(minimumLeafSize), 1.0, 1e-5);
  }
}

/**
 * Test k-fold cross-validation with the MSE metric.
 */
BOOST_AUTO_TEST_CASE(KFoldCVMSETest)
{
  // Defining dataset with two sets of responses for the same two data points.
  arma::mat data("0 1  0 1");
  arma::rowvec responses("0 1  1 3");

  // 2-fold cross-validation, no shuffling.
  KFoldCV<LinearRegression, MSE> cv(2, data, responses, false);

  // In each of two validation tests the MSE value should be the same.
  double expectedMSE =
      double((1 - 0) * (1 - 0) + (3 - 1) * (3 - 1)) / 2 * 2 / 2;

  BOOST_REQUIRE_CLOSE(cv.Evaluate(), expectedMSE, 1e-5);

  // Assert we can access a trained model without the exception of
  // uninitialization.
  cv.Model();
}

/**
 * Test k-fold cross-validation with the Accuracy metric.
 */
BOOST_AUTO_TEST_CASE(KFoldCVAccuracyTest)
{
  // Making a 10-points dataset. The last point should be classified wrong when
  // it is tested separately.
  arma::mat data("0 1 2 3 100 101 102 103 104 5");
  arma::Row<size_t> labels("0 0 0 0 1 1 1 1 1 1");
  size_t numClasses = 2;

  // 10-fold cross-validation, no shuffling.
  KFoldCV<NaiveBayesClassifier<>, Accuracy> cv(10, data, labels, numClasses,
      false);

  // We should succeed in classifying separately the first nine samples, and
  // fail with the remaining one.
  double expectedAccuracy = (9 * 1.0 + 0.0) / 10;

  BOOST_REQUIRE_CLOSE(cv.Evaluate(), expectedAccuracy, 1e-5);

  // Assert we can access a trained model without the exception of
  // uninitialization.
  cv.Model();
}

/**
 * Test k-fold cross-validation with weighted linear regression.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithWeightedLRTest)
{
  // Each fold will be filled with this dataset.
  arma::mat data("1 2 3 4");
  arma::rowvec responses("1 2 30 40");
  arma::rowvec weights("1 1 0 0");

  KFoldCV<LinearRegression, MSE> cv(2, arma::join_rows(data, data),
      arma::join_rows(responses, responses), arma::join_rows(weights, weights),
      false);
  cv.Evaluate();

  arma::mat testData("3 4");
  arma::rowvec testResponses("3 4");

  double mse = MSE::Evaluate(cv.Model(), testData, testResponses);

  BOOST_REQUIRE_CLOSE(1.0 - mse, 1.0, 1e-5);
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTest)
{
  arma::mat originalData;
  arma::Row<size_t> originalLabels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(originalData, originalLabels, datasetInfo);

  // Each fold will be filled with this dataset.
  arma::mat data = originalData.cols(0, 1199);
  arma::Row<size_t> labels = originalLabels.cols(0, 1199);
  arma::rowvec weights(data.n_cols, arma::fill::randu);

  arma::mat doubledData = arma::join_rows(data, data);
  arma::Row<size_t> doubledLabels = arma::join_rows(labels, labels);
  arma::rowvec doubledWeights = arma::join_rows(weights, weights);

  size_t numClasses = 5;
  size_t minimumLeafSize = 8;

  {
    KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(2, doubledData,
        doubledLabels, numClasses, false);
    cv.Evaluate(minimumLeafSize);
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(data, data, labels,
        numClasses, minimumLeafSize);
    double accuracy = Accuracy::Evaluate(cv.Model(), data, predictedLabels);
    BOOST_REQUIRE_CLOSE(accuracy, 1.0, 1e-5);
  }
  {
    KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(2, doubledData,
        datasetInfo, doubledLabels, numClasses, false);
    cv.Evaluate(minimumLeafSize);
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(data, data,
        datasetInfo, labels, numClasses, minimumLeafSize);
    double accuracy = Accuracy::Evaluate(cv.Model(), data, predictedLabels);
    BOOST_REQUIRE_CLOSE(accuracy, 1.0, 1e-5);
  }
  {
    KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(2, doubledData,
        doubledLabels, numClasses, doubledWeights, false);
    cv.Evaluate(minimumLeafSize);
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(data, data, labels,
        numClasses, weights, minimumLeafSize);
    double accuracy = Accuracy::Evaluate(cv.Model(), data, predictedLabels);
    BOOST_REQUIRE_CLOSE(accuracy, 1.0, 1e-5);
  }
  {
    KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(2, doubledData,
        datasetInfo, doubledLabels, numClasses, doubledWeights, false);
    cv.Evaluate(minimumLeafSize);
    arma::Row<size_t> predictedLabels = PredictLabelsWithDT(data, data,
        datasetInfo, labels, numClasses, weights, minimumLeafSize);
    double accuracy = Accuracy::Evaluate(cv.Model(), data, predictedLabels);
    BOOST_REQUIRE_CLOSE(accuracy, 1.0, 1e-5);
  }
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways, but with larger k and no shuffling.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTestLargeKNoShuffle)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  size_t numClasses = 5;
  size_t minimumLeafSize = 5;

  KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(5, data,
      datasetInfo, labels, numClasses, false);
  cv.Evaluate(minimumLeafSize);
  double accuracy = Accuracy::Evaluate(cv.Model(), data, labels);

  // This is a very loose tolerance, but we expect about the same as we would
  // from an individual decision tree training.
  BOOST_REQUIRE_GT(accuracy, 0.7);
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways, but with larger k such that the number of points in each
 * cross-validation bin is not even (the last is smaller), and also with no
 * shuffling.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTestUnevenBinsNoShuffle)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  size_t numClasses = 5;
  size_t minimumLeafSize = 5;

  KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(7, data, datasetInfo,
      labels, numClasses, false);
  cv.Evaluate(minimumLeafSize);
  double accuracy = Accuracy::Evaluate(cv.Model(), data, labels);

  // This is a very loose tolerance, but we expect about the same as we would
  // from an individual decision tree training.
  BOOST_REQUIRE_GT(accuracy, 0.7);
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways, but with larger k.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTestLargeK)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  size_t numClasses = 5;
  size_t minimumLeafSize = 5;

  KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(5, data,
      datasetInfo, labels, numClasses);
  cv.Evaluate(minimumLeafSize);
  double accuracy = Accuracy::Evaluate(cv.Model(), data, labels);

  // This is a very loose tolerance, but we expect about the same as we would
  // from an individual decision tree training.
  BOOST_REQUIRE_GT(accuracy, 0.7);
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways, but with larger k such that the number of points in each
 * cross-validation bin is not even (the last is smaller).
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTestUnevenBins)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);

  size_t numClasses = 5;
  size_t minimumLeafSize = 5;

  KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(7, data, datasetInfo,
      labels, numClasses);
  cv.Evaluate(minimumLeafSize);
  double accuracy = Accuracy::Evaluate(cv.Model(), data, labels);

  // This is a very loose tolerance, but we expect about the same as we would
  // from an individual decision tree training.
  BOOST_REQUIRE_GT(accuracy, 0.7);
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways, but with larger k and weights.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTestLargeKWeighted)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);
  arma::rowvec weights(data.n_cols, arma::fill::randu);

  size_t numClasses = 5;
  size_t minimumLeafSize = 5;

  KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(5, data,
      datasetInfo, labels, numClasses, weights);
  cv.Evaluate(minimumLeafSize);
  double accuracy = Accuracy::Evaluate(cv.Model(), data, labels);

  // This is a very loose tolerance, but we expect about the same as we would
  // from an individual decision tree training.
  BOOST_REQUIRE_GT(accuracy, 0.7);
}

/**
 * Test k-fold cross-validation with decision trees constructed in multiple
 * ways, but with larger k such that the number of points in each
 * cross-validation bin is not even (the last is smaller) and weights.
 */
BOOST_AUTO_TEST_CASE(KFoldCVWithDTTestUnevenBinsWeighted)
{
  arma::mat data;
  arma::Row<size_t> labels;
  data::DatasetInfo datasetInfo;
  MockCategoricalData(data, labels, datasetInfo);
  arma::rowvec weights(data.n_cols, arma::fill::randu);

  size_t numClasses = 5;
  size_t minimumLeafSize = 5;

  KFoldCV<DecisionTree<InformationGain>, Accuracy> cv(7, data, datasetInfo,
      labels, numClasses, weights);
  cv.Evaluate(minimumLeafSize);
  double accuracy = Accuracy::Evaluate(cv.Model(), data, labels);

  // This is a very loose tolerance, but we expect about the same as we would
  // from an individual decision tree training.
  BOOST_REQUIRE_GT(accuracy, 0.7);
}

BOOST_AUTO_TEST_SUITE_END();
