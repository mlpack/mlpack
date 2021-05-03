/**
 * @file tests/decision_tree_regressor_test.cpp
 * @author Rishabh Garg
 *
 * Tests for the DecisionTreeRegressor class and related classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <mlpack/methods/decision_tree/mad_gain.hpp>
#include <mlpack/methods/decision_tree/mse_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::distribution;

/**
 * Make sure the MSE gain is zero when the labels are perfect.
 */
TEST_CASE("MSEGainPerfectTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec labels;
  labels.ones(10);

  REQUIRE(MSEGain::Evaluate<false>(labels, 0, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure that the MSE gain is equal to negative of variance.
 */
TEST_CASE("MSEGainVarianceTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights(100, arma::fill::ones);
  arma::rowvec labels(100, arma::fill::randn);

  // Theoretical gain.
  double theoreticalGain = - arma::var(labels) * 99.0 / 100.0;

  // Calculated gain.
  const double calculatedGain = MSEGain::Evaluate<false>(labels, 0, weights);

  REQUIRE(calculatedGain == Approx(theoreticalGain).margin(1e-9));
}

/**
 * The MSE gain of an empty vector is 0.
 */
TEST_CASE("MSEGainEmptyTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec labels;
  REQUIRE(MSEGain::Evaluate<false>(labels, 0, weights) ==
          Approx(0.0).margin(1e-5));

  REQUIRE(MSEGain::Evaluate<true>(labels, 0, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure the MAD gain is zero when the labels are perfect.
 */
TEST_CASE("MADGainPerfectTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec labels;
  labels.ones(10);

  REQUIRE(MADGain::Evaluate<false>(labels, 0, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure that when mean of labels is zero, MAD_gain = mean of
 * absolute values of the distribution.
 */
TEST_CASE("MADGainNormalTest", "[DecisionTreeRegressorTest")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec labels = { 1, 2, 3, 4, 5, -1, -2, -3, -4, -5 }; // Mean = 0.

  // Theoretical gain.
  double theoreticalGain = 0.0;
  for (size_t i = 0; i < labels.n_elem; ++i)
    theoreticalGain -= std::abs(labels[i]);
  theoreticalGain /= (double) labels.n_elem;

  // Calculated gain.
  const double calculatedGain = MADGain::Evaluate<false>(labels, 0, weights);

  REQUIRE(calculatedGain == Approx(theoreticalGain).margin(1e-5));
}

/**
 * The MAD gain of an empty vector is 0.
 */
TEST_CASE("MADGainEmptyTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec labels;
  REQUIRE(MADGain::Evaluate<false>(labels, 0, weights) ==
          Approx(0.0).margin(1e-5));

  REQUIRE(MADGain::Evaluate<true>(labels, 0, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Check that AllCategoricalSplit will split when the split is obviously
 * better.
 */
TEST_CASE("AllCategoricalSplitSimpleSplitTest1", "[DecisionTreeRegressorTest]")
{
  arma::vec predictor(100);
  arma::rowvec labels(100);
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  for (size_t i = 0; i < 100; i+=2)
  {
    predictor[i] = 0;
    labels[i] = 5.0;
    predictor[i + 1] = 1;
    labels[i + 1] = 100;
  }

  double splitInfo;
  AllCategoricalSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 0, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictor, 2, labels, 0, weights, 3, 1e-7, splitInfo, aux);
  const double weightedGain =
      AllCategoricalSplit<MSEGain>::SplitIfBetter<true>(bestGain, predictor, 2,
      labels, 0, weights, 3, 1e-7, splitInfo, aux);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  REQUIRE(gain == weightedGain);

  // Make sure that splitInfo now hold the number of children.
  REQUIRE((size_t) splitInfo == 2);
}

/**
 * Make sure that AllCategoricalSplit respects the minimum number of samples
 * required to split.
 */
TEST_CASE("AllCategoricalSplitMinSamplesTest1", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  arma::rowvec labels = {0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2};
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  double splitInfo;
  AllCategoricalSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 0, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, 4, labels, 0, weights, 4, 1e-7, splitInfo, aux);

  // Make sure it's not split.
  REQUIRE(gain == DBL_MAX);
}

/**
 * Check that no split is made when it doesn't get us anything.
 */
TEST_CASE("AllCategoricalSplitNoGainTest1", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors(300);
  arma::rowvec labels(300);
  arma::rowvec weights = arma::ones<arma::rowvec>(300);

  for (size_t i = 0; i < 300; i += 3)
  {
    predictors[i] = int(i / 3) % 10;
    labels[i] = -0.5;
    predictors[i + 1] = int(i / 3) % 10;
    labels[i + 1] = 0;
    predictors[i + 2] = int(i / 3) % 10;
    labels[i + 2] = 0.5;
  }

  double splitInfo;
  AllCategoricalSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 0, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, 10, labels, 0, weights, 10, 1e-7,
      splitInfo, aux);
  const double weightedGain =
      AllCategoricalSplit<MSEGain>::SplitIfBetter<true>(bestGain, predictors,
      10, labels, 0, predictors, 10, 1e-7, splitInfo, aux);

  // Make sure that there was no split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the BestBinaryNumericSplit will split on an obviously splittable
 * dimension.
 */
TEST_CASE("BestBinaryNumericSplitSimpleSplitTest1", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
  arma::rowvec labels = { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  double splitInfo;
  BestBinaryNumericSplit<MADGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MADGain::Evaluate<false>(labels, 0, weights);
  const double gain = BestBinaryNumericSplit<MADGain>::SplitIfBetter<false>(
      bestGain, predictors, labels, 0, weights, 3, 1e-7, splitInfo,
      aux);
  const double weightedGain =
      BestBinaryNumericSplit<MADGain>::SplitIfBetter<true>(bestGain, predictors,
      labels, 0, weights, 3, 1e-7, splitInfo, aux);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Make sure weight works and is not different than the unweighted one.
  REQUIRE(gain == weightedGain);

  // The class probabilities, for this split, hold the splitting point, which
  // should be between 4 and 5.
  REQUIRE(splitInfo > 0.4);
  REQUIRE(splitInfo < 0.5);
  std::cout << "Done\n";
}

/**
 * Check that the BestBinaryNumericSplit won't split if not enough points are
 * given.
 */
TEST_CASE("BestBinaryNumericSplitMinSamplesTest1", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
  arma::rowvec labels = { 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  arma::rowvec weights(labels.n_elem);

  double splitInfo;
  BestBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 0, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, labels, 0, weights, 8, 1e-7, splitInfo, aux);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      BestBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, predictors,
      labels, 0, weights, 8, 1e-7, splitInfo, aux);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the BestBinaryNumericSplit doesn't split a dimension that gives no
 * gain.
 */
TEST_CASE("BestBinaryNumericSplitNoGainTest1", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors(100);
  arma::rowvec labels(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    predictors[i] = i;
    labels[i] = 0.0;
    predictors[i + 1] = i;
    labels[i + 1] = 1.0;
  }

  double splitInfo;
  BestBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 0, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, labels, 0, weights, 10, 1e-7, splitInfo,
      aux);

  // Make sure there was no split.
  REQUIRE(gain == DBL_MAX);
}
