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
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "mock_categorical_data.hpp"

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::cv;
using namespace mlpack::naive_bayes;
using namespace mlpack::optimization;
using namespace mlpack::regression;
using namespace mlpack::tree;

BOOST_AUTO_TEST_SUITE(CVTest);

/*
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

/*
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

/*
 * Test the mean squared error with matrix responses.
 */
BOOST_AUTO_TEST_CASE(MSEMatResponsesTest)
{
  arma::mat data("1 2");
  arma::mat trainingResponses("1 2; 3 4");

  FFN<MeanSquaredError<>, ZeroInitialization> ffn;
  ffn.Add<Linear<>>(1, 2);
  ffn.Add<IdentityLayer<>>();

  RMSProp opt(0.2);
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

BOOST_AUTO_TEST_CASE(WeightsTypeTest)
{
  CheckWeightsType<LinearRegression, arma::rowvec>();
  CheckWeightsType<DecisionTree<>, arma::rowvec>();
  CheckWeightsType<DecisionTree<>, arma::Row<float>, arma::mat,
      arma::Row<size_t>, arma::Row<float>>();
}

BOOST_AUTO_TEST_CASE(TakesDatasetInfoTest)
{
  static_assert(MetaInfoExtractor<DecisionTree<>>::TakesDatasetInfo,
      "Value should be true");
  static_assert(!MetaInfoExtractor<LinearRegression>::TakesDatasetInfo,
      "Value should be false");
  static_assert(!MetaInfoExtractor<SoftmaxRegression>::TakesDatasetInfo,
      "Value should be false");
}

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

  size_t numClasses = 3;
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

BOOST_AUTO_TEST_SUITE_END();
