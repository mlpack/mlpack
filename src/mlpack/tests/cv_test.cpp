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
#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <mlpack/methods/naive_bayes/naive_bayes_classifier.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <boost/test/unit_test.hpp>

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

  CheckWeightsType<FFN<>, void>();
  CheckWeightsType<LARS, void>();
  CheckWeightsType<LogisticRegression<>, void>();
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

BOOST_AUTO_TEST_SUITE_END();
