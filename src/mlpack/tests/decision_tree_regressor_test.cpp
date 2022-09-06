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
#include <mlpack/methods/decision_tree.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;

/**
 * Creates dataset with 5 groups with all the points in same group have exactly
 * same responses.
 */
void CreateMultiSplitData(arma::mat& d, arma::rowvec& r, const size_t count,
    arma::rowvec& values)
{
  d = arma::mat(10, count, arma::fill::randu);
  r = arma::rowvec(count);

  // Group 1.
  for (size_t i = 0; i < count / 5; i++)
  {
    d(3, i) = i;
    r(i) = values[0];
  }
  // Group 2.
  for (size_t i = count / 5; i < (count / 5) * 2; i++)
  {
    d(3, i) = i;
    r(i) = values[1];
  }
  // Group 3.
  for (size_t i = (count / 5) * 2; i < (count / 5) * 3; i++)
  {
    d(3, i) = i;
    r(i) = values[2];
  }
  // Group 4.
  for (size_t i = (count / 5) * 3; i < (count / 5) * 4; i++)
  {
    d(3, i) = i;
    r(i) = values[3];
  }
  // Group 5.
  for (size_t i = (count / 5) * 4; i < count; i++)
  {
    d(3, i) = i;
    r(i) = values[4];
  }
}

/**
 * Make sure the MSE gain is zero when the responses are perfect.
 */
TEST_CASE("MSEGainPerfectTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec responses;
  responses.ones(10);

  REQUIRE(MSEGain::Evaluate<false>(responses, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * The MSE gain of an empty vector is 0.
 */
TEST_CASE("MSEGainEmptyTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec responses;

  REQUIRE(MSEGain::Evaluate<false>(responses, weights) ==
          Approx(0.0).margin(1e-5));
  REQUIRE(MSEGain::Evaluate<true>(responses, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Making sure that MSE gain is evaluated correctly by doing calculation by
 * hand.
 */
TEST_CASE("MSEGainHandCalculation", "[DecisionTreeRegressorTest]")
{
  arma::rowvec responses = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  arma::rowvec weights = {0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7};

  // Hand calculated gain values.
  const double gain = -27.08999;
  const double weightedGain = -27.53960;

  REQUIRE(MSEGain::Evaluate<false>(responses, weights) ==
          Approx(gain).margin(1e-5));
  REQUIRE(MSEGain::Evaluate<true>(responses, weights) ==
          Approx(weightedGain).margin(1e-5));
}

/**
 * Make sure the MAD gain is zero when the responses are perfect.
 */
TEST_CASE("MADGainPerfectTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec responses;
  responses.ones(10);

  REQUIRE(MADGain::Evaluate<false>(responses, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Make sure that when mean of responses is zero, MAD_gain = mean of
 * absolute values of the distribution.
 */
TEST_CASE("MADGainNormalTest", "[DecisionTreeRegressorTest")
{
  arma::rowvec weights(10, arma::fill::ones);
  arma::rowvec responses = { 1, 2, 3, 4, 5, -1, -2, -3, -4, -5 }; // Mean = 0.

  // Theoretical gain.
  double theoreticalGain = 0.0;
  for (size_t i = 0; i < responses.n_elem; ++i)
    theoreticalGain -= std::abs(responses[i]);
  theoreticalGain /= (double) responses.n_elem;

  // Calculated gain.
  const double calculatedGain = MADGain::Evaluate<false>(responses, weights);

  REQUIRE(calculatedGain == Approx(theoreticalGain).margin(1e-5));
}

/**
 * The MAD gain of an empty vector is 0.
 */
TEST_CASE("MADGainEmptyTest", "[DecisionTreeRegressorTest]")
{
  arma::rowvec weights = arma::ones<arma::rowvec>(10);
  arma::rowvec responses;

  REQUIRE(MADGain::Evaluate<false>(responses, weights) ==
          Approx(0.0).margin(1e-5));
  REQUIRE(MADGain::Evaluate<true>(responses, weights) ==
          Approx(0.0).margin(1e-5));
}

/**
 * Making sure that MAD gain is evaluated correctly by doing calculation by
 * hand.
 */
TEST_CASE("MADGainHandCalculation", "[DecisionTreeRegressorTest]")
{
  arma::rowvec responses = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  arma::rowvec weights = {0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7};

  // Hand calculated gain values.
  const double gain = -4.1;
  const double weightedGain = -3.8592;

  REQUIRE(MADGain::Evaluate<false>(responses, weights) ==
          Approx(gain).margin(1e-5));
  REQUIRE(MADGain::Evaluate<true>(responses, weights) ==
          Approx(weightedGain).margin(1e-5));
}

/**
 * Check that AllCategoricalSplit will split when the split is obviously
 * better.
 */
TEST_CASE("AllCategoricalSplitSimpleSplitTest_", "[DecisionTreeRegressorTest]")
{
  arma::vec predictor(100);
  arma::rowvec responses(100);
  arma::rowvec weights(responses.n_elem);
  weights.ones();

  for (size_t i = 0; i < 100; i+=2)
  {
    predictor[i] = 0;
    responses[i] = 5.0;
    predictor[i + 1] = 1;
    responses[i + 1] = 100;
  }

  double splitInfo;
  AllCategoricalSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictor, 2, responses, weights, 3, 1e-7, splitInfo, aux, f);
  const double weightedGain =
      AllCategoricalSplit<MSEGain>::SplitIfBetter<true>(bestGain, predictor, 2,
      responses, weights, 3, 1e-7, splitInfo, aux, f);

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
TEST_CASE("AllCategoricalSplitMinSamplesTest_", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  arma::rowvec responses = {0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2};
  arma::rowvec weights(responses.n_elem);
  weights.ones();

  double splitInfo;
  AllCategoricalSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, 4, responses, weights, 4, 1e-7, splitInfo, aux, f);

  // Make sure it's not split.
  REQUIRE(gain == DBL_MAX);
}

/**
 * Check that no split is made when it doesn't get us anything.
 */
TEST_CASE("AllCategoricalSplitNoGainTest_", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors(300);
  arma::rowvec responses(300);
  arma::rowvec weights = arma::ones<arma::rowvec>(300);

  for (size_t i = 0; i < 300; i += 3)
  {
    predictors[i] = int(i / 3) % 10;
    responses[i] = -0.5;
    predictors[i + 1] = int(i / 3) % 10;
    responses[i + 1] = 0;
    predictors[i + 2] = int(i / 3) % 10;
    responses[i + 2] = 0.5;
  }

  double splitInfo;
  AllCategoricalSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = AllCategoricalSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, 10, responses, weights, 10, 1e-7,
      splitInfo, aux, f);
  const double weightedGain =
      AllCategoricalSplit<MSEGain>::SplitIfBetter<true>(bestGain, predictors,
      10, responses, weights, 10, 1e-7, splitInfo, aux, f);

  // Make sure that there was no split.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the BestBinaryNumericSplit will split on an obviously splittable
 * dimension.
 */
TEST_CASE("BestBinaryNumericSplitSimpleSplitTest_",
    "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors =
      { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
  arma::rowvec responses =
      { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  arma::rowvec weights(responses.n_elem);
  weights.ones();

  double splitInfo;
  BestBinaryNumericSplit<MADGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MADGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = BestBinaryNumericSplit<MADGain>::SplitIfBetter<false>(
      bestGain, predictors, responses, weights, 3, 1e-7, splitInfo, aux, f);
  const double weightedGain =
      BestBinaryNumericSplit<MADGain>::SplitIfBetter<true>(bestGain, predictors,
      responses, weights, 3, 1e-7, splitInfo, aux, f);

  // Make sure that a split was made.
  REQUIRE(gain > bestGain);

  // Make sure weight works and is not different than the unweighted one.
  REQUIRE(gain == weightedGain);

  // The class probabilities, for this split, hold the splitting point, which
  // should be between 4 and 5.
  REQUIRE(splitInfo > 0.4);
  REQUIRE(splitInfo < 0.5);
}

/**
 * Check that the BestBinaryNumericSplit won't split if not enough points are
 * given.
 */
TEST_CASE("BestBinaryNumericSplitMinSamplesTest_",
    "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors =
      { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
  arma::rowvec responses =
      { 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  arma::rowvec weights(responses.n_elem);

  double splitInfo;
  BestBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, responses, weights, 8, 1e-7, splitInfo, aux, f);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      BestBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain,
      predictors, responses, weights, 8, 1e-7, splitInfo, aux, f);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the BestBinaryNumericSplit doesn't split a dimension that gives
 * no gain.
 */
TEST_CASE("BestBinaryNumericSplitNoGainTest_", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors(100);
  arma::rowvec responses(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    predictors[i] = i;
    responses[i] = 0.0;
    predictors[i + 1] = i;
    responses[i + 1] = 1.0;
  }

  double splitInfo;
  BestBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, responses, weights, 10, 1e-7, splitInfo, aux, f);

  // Make sure there was no split.
  REQUIRE(gain == DBL_MAX);
}

/**
 * Check that the RandomBinaryNumericSplit always splits when splitIfBetterGain
 * is false.
 */
TEST_CASE("RandomBinaryNumericSplitAlwaysSplit_",
    "[DecisionTreeRegressorTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::rowvec responses("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights;
  weights.ones(responses.n_elem);

  double splitInfo;
  RandomBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, responses, weights, 1, 1e-7, splitInfo, aux, f);
  const double weightedGain =
      RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, values,
      responses, weights, 1, 1e-7, splitInfo, aux, f);

  // Make sure that split was made.
  REQUIRE(gain != DBL_MAX);
  REQUIRE(weightedGain != DBL_MAX);
}

/**
 * Check that the RandomBinaryNumericSplit won't split if not enough points are
 * given.
 */
TEST_CASE("RandomBinaryNumericSplitMinSamplesTest_",
    "[DecisionTreeRegressorTest]")
{
  arma::vec values("0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0");
  arma::rowvec responses("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(responses.n_elem);

  double splitInfo;
  RandomBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, responses, weights, 8, 1e-7, splitInfo, aux, f);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, values,
      responses, weights, 8, 1e-7, splitInfo, aux, f);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the RandomBinaryNumericSplit doesn't split a dimension that gives
 * no gain when splitIfBetterGain is true.
 */
TEST_CASE("RandomBinaryNumericSplitNoGainTest_", "[DecisionTreeRegressorTest]")
{
  arma::vec values(100);
  arma::rowvec responses(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    values[i] = i;
    responses[i] = 0.0;
    values[i + 1] = i;
    responses[i + 1] = 1.0;
  }

  double splitInfo;
  RandomBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  MSEGain f;
  const double bestGain = f.Evaluate<false>(responses, weights);
  const double gain = RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, responses, weights, 10, 1e-7, splitInfo, aux, f, true);

  // Make sure there was no split.
  REQUIRE(gain == DBL_MAX);
}

/**
 * A basic construction of the decision tree---ensure that we can create the
 * tree and that it split at least once.
 */
TEST_CASE("BasicConstructionTest_", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }

  // Use default parameters.
  DecisionTreeRegressor<> d(dataset, responses);

  // Now require that we have some children.
  REQUIRE(d.NumChildren() > 0);
}

/**
 * Construct a tree with weighted responses.
 */
TEST_CASE("BasicConstructionTestWithWeight_", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }
  arma::rowvec weights(responses.n_elem);
  weights.ones();

  // Use default parameters.
  DecisionTreeRegressor<> wd(dataset, responses, weights);
  DecisionTreeRegressor<> d(dataset, responses);

  // Now require that we have some children.
  REQUIRE(wd.NumChildren() > 0);
  REQUIRE(wd.NumChildren() == d.NumChildren());
}

/**
 * Construct the decision tree on numeric data only and see that we can fit it
 * exactly and achieve perfect performance on the training set.
 */
TEST_CASE("PerfectTrainingSet_", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, responses, 1, 0.0);

  // Make sure that we can get perfect fit on the training set.
  for (size_t i = 0; i < 100; ++i)
  {
    double prediction;
    prediction = d.Predict(dataset.col(i));

    REQUIRE(prediction == Approx(responses[i]).epsilon(1e-7));
  }
}

/**
 * Construct the decision tree with weighted responses.
 */
TEST_CASE("PerfectTrainingSetWithWeight_", "[DecisionTreeRegressorTest]")
{
  // Completely random dataset with no structure.
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::rowvec responses(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    responses[i] = 1.0;
  }
  arma::rowvec weights = arma::ones<arma::rowvec>(responses.n_elem);

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, responses, weights, 1, 0.0);

  // This part of code is dupliacte with no weighted one.
  for (size_t i = 0; i < 100; ++i)
  {
    size_t prediction;
    prediction = d.Predict(dataset.col(i));

    REQUIRE(prediction == Approx(responses[i]).epsilon(1e-7));
  }
}

/**
 * Test that the tree is able to perfectly fit all the obvious splits present
 * in the data.
 *
 *     |
 *     |
 *   2 |            xxxxxx
 *     |
 *     |
 *   1 |      xxxxxx      xxxxxx
 *     |
 *     |
 *   0 |xxxxxx                  xxxxxx
 *     |___________________________________
 */
TEST_CASE("MultiSplitTest1", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset;
  arma::rowvec responses;
  arma::rowvec values = {0.0, 1.0, 2.0, 1.0, 0.0};

  CreateMultiSplitData(dataset, responses, 1000, values);

  arma::rowvec weights(responses.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, responses, weights, 2, 0.0);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  // Ensure that the predictions are perfect.
  for (size_t i = 0; i < responses.n_elem; ++i)
    REQUIRE(preds[i] == responses[i]);

  // Ensure that a split is made only when required and no redundant splits are
  // made.
  REQUIRE(d.NumLeaves() == 5);
}

/**
 * Test that the tree is able to perfectly fit all the obvious splits present
 * in the data. Same test as above, but with less data.
 */
TEST_CASE("MultiSplitTest2", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset;
  arma::rowvec responses;
  arma::rowvec values = {0.0, 1.0, 2.0, 1.0, 0.0};

  CreateMultiSplitData(dataset, responses, 1000, values);

  arma::rowvec weights(responses.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, responses, weights, 2, 0.0);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  // Ensure that the predictions are perfect.
  for (size_t i = 0; i < responses.n_elem; ++i)
    REQUIRE(preds[i] == responses[i]);

  // Ensure that a split is made only when required and no redundant splits are
  // made.
  REQUIRE(d.NumLeaves() == 5);
}

/**
 * Test that the tree is able to perfectly fit all the obvious splits present
 * in the data.
 *
 *     |
 *  20 |                        xxxxxx
 *     |
 *     |
 *  15 |                  xxxxxx
 *     |
 *     |
 *  10 |            xxxxxx
 *     |
 *     |
 *   5 |      xxxxxx
 *     |
 *     |
 *   0 |xxxxxx
 *     |________________________________________
 */
TEST_CASE("MultiSplitTest3", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset;
  arma::Row<double> responses;
  arma::rowvec values = {0.0, 5.0, 10.0, 15.0, 20.0};

  CreateMultiSplitData(dataset, responses, 500, values);

  arma::rowvec weights(responses.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, responses, weights, 2, 0.0);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  // Ensure that the predictions are perfect.
  for (size_t i = 0; i < responses.n_elem; ++i)
    REQUIRE(preds[i] == responses[i]);

  // Ensure that a split is made only when required and no redundant splits are
  // made.
  REQUIRE(d.NumLeaves() == 5);
}

/**
 * Test that the tree builds correctly on unweighted numerical dataset.
 */
TEST_CASE("NumericalBuildTest", "[DecisionTreeRegressorTest]")
{
  arma::mat X;
  arma::rowvec Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::mat XTrain, XTest;
  arma::rowvec YTrain, YTest;
  data::Split(X, Y, XTrain, XTest, YTrain, YTest, 0.3);

  DecisionTreeRegressor<> tree(XTrain, YTrain, 5);

  arma::rowvec predictions;
  tree.Predict(XTest, predictions);

  // Ensuring a decent performance.
  const double rmse = RMSE(predictions, YTest);
  REQUIRE(rmse < 1.0);
}

/**
 * Test that the tree builds correctly on weighted numerical dataset.
 */
TEST_CASE("NumericalBuildTestWithWeights", "[DecisionTreeRegressorTest]")
{
  arma::mat X;
  arma::rowvec Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::mat XTrain, XTest;
  arma::rowvec YTrain, YTest;
  data::Split(X, Y, XTrain, XTest, YTrain, YTest, 0.3);

  arma::rowvec weights = arma::ones<arma::rowvec>(XTrain.n_elem);

  DecisionTreeRegressor<> tree(XTrain, YTrain, weights, 5);

  arma::rowvec predictions;
  tree.Predict(XTest, predictions);

  // Ensuring a decent performance.
  const double rmse = RMSE(predictions, YTest);
  REQUIRE(rmse < 1.0);
}

/**
 * Test that we can build a decision tree on a simple categorical dataset.
 */
TEST_CASE("CategoricalBuildTest_", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec r;
  data::DatasetInfo di;
  MockCategoricalData(d, r, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingResponses = r.subvec(0, 1999);
  arma::rowvec testResponses = r.subvec(2000, 3999);

  // Build the tree.
  DecisionTreeRegressor<> tree(trainingData, di, trainingResponses, 10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 1.0);
}

/**
 * Test that we can build a decision tree with weights on a simple categorical
 * dataset.
 */
TEST_CASE("CategoricalBuildTestWithWeight_", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec r;
  data::DatasetInfo di;
  MockCategoricalData(d, r, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingResponses = r.subvec(0, 1999);
  arma::rowvec testResponses = r.subvec(2000, 3999);

  arma::rowvec weights = arma::ones<arma::rowvec>(trainingResponses.n_elem);

  // Build the tree.
  DecisionTreeRegressor<> tree(trainingData, di, trainingResponses, weights,
      10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 1.0);
}

/**
 * Test that we can build a decision tree on a simple categorical dataset using
 * weights, with low-weight noise added.
 */
TEST_CASE("CategoricalWeightedBuildTest_", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec r;
  data::DatasetInfo di;
  MockCategoricalData(d, r, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingResponses = r.subvec(0, 1999);
  arma::rowvec testResponses = r.subvec(2000, 3999);

  // Now create random points.
  arma::mat randomNoise(5, 2000);
  arma::rowvec randomResponses(2000);
  for (size_t i = 0; i < 2000; ++i)
  {
    randomNoise(0, i) = Random();
    randomNoise(1, i) = Random(-1, 1);
    randomNoise(2, i) = Random();
    randomNoise(3, i) = RandInt(0, 2);
    randomNoise(4, i) = RandInt(0, 5);
    randomResponses[i] = Random(-10, 18);
  }

  // Generate weights.
  arma::rowvec weights(4000);
  for (size_t i = 0; i < 2000; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = 2000; i < 4000; ++i)
    weights[i] = Random(0.0, 0.001);

  arma::mat fullData = arma::join_rows(trainingData, randomNoise);
  arma::rowvec fullResponses = arma::join_rows(trainingResponses,
      randomResponses);

  // Build the tree.
  DecisionTreeRegressor<> tree(fullData, di, fullResponses, weights, 10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 1.5);
}

/**
 * Test that we can build a decision tree using MAD gain on a simple
 * categorical dataset using weights, with low-weight noise added.
 */
TEST_CASE("CategoricalMADGainWeightedBuildTest", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec r;
  data::DatasetInfo di;
  MockCategoricalData(d, r, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingResponses = r.subvec(0, 1999);
  arma::rowvec testResponses = r.subvec(2000, 3999);

  // Now create random points.
  arma::mat randomNoise(5, 2000);
  arma::rowvec randomResponses(2000);
  for (size_t i = 0; i < 2000; ++i)
  {
    randomNoise(0, i) = Random();
    randomNoise(1, i) = Random(-1, 1);
    randomNoise(2, i) = Random();
    randomNoise(3, i) = RandInt(0, 2);
    randomNoise(4, i) = RandInt(0, 5);
    randomResponses[i] = Random(-10, 18);
  }

  // Generate weights.
  arma::rowvec weights(4000);
  for (size_t i = 0; i < 2000; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = 2000; i < 4000; ++i)
    weights[i] = Random(0.0, 0.001);

  arma::mat fullData = arma::join_rows(trainingData, randomNoise);
  arma::rowvec fullResponses = arma::join_rows(trainingResponses,
      randomResponses);

  // Build the tree using MAD gain.
  DecisionTreeRegressor<MADGain> tree(fullData, di, fullResponses, weights,
      10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 1.05);
}

/**
 * Test that the decision tree generalizes reasonably.
 */
TEST_CASE("SimpleGeneralizationTest_", "[DecisionTreeRegressorTest]")
{
  // Loading data.
  data::DatasetInfo info;
  arma::mat trainData, testData;
  arma::rowvec trainResponses, testResponses;
  LoadBostonHousingDataset(trainData, testData, trainResponses, testResponses,
      info);
  arma::rowvec weights = arma::ones<arma::rowvec>(trainResponses.n_elem);

  // Build decision tree.
  DecisionTreeRegressor<> d(trainData, info, trainResponses);
  DecisionTreeRegressor<> wd(trainData, info, trainResponses, weights);

  // Get the predicted test responses.
  arma::rowvec predictions;
  d.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out rmse.
  double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 6.1);

  // Reset the predictions.
  predictions.zeros();
  wd.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out rmse.
  rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 6.1);
}

/**
 * Test that the decision tree generalizes reasonably when built on float data.
 */
TEST_CASE("SimpleGeneralizationFMatTest_", "[DecisionTreeRegressorTest]")
{
  // Loading data.
  data::DatasetInfo info;
  arma::fmat trainData, testData;
  arma::rowvec trainLabels, testLabels;
  LoadBostonHousingDataset(trainData, testData, trainLabels, testLabels, info);

  // Initialize an all-ones weight matrix.
  arma::rowvec weights(trainLabels.n_cols, arma::fill::ones);

  // Build decision tree.
  DecisionTreeRegressor<> d(trainData, trainLabels);
  DecisionTreeRegressor<> wd(trainData, trainLabels, weights);

  // Get the predicted test labels.
  arma::rowvec predictions;
  d.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the rmse.
  double rmse = RMSE(predictions, testLabels);
  REQUIRE(rmse < 6.0);

  // Reset the prediction.
  predictions.zeros();
  wd.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the rmse.
  double wdrmse = RMSE(predictions, testLabels);
  REQUIRE(wdrmse < 6.0);
}

/**
 * Test that we can build a decision tree using weighted data (where the
 * low-weighted data is random noise), and that the tree still builds correctly
 * enough to get good results.
 */
TEST_CASE("WeightedDecisionTreeTest_", "[DecisionTreeRegressorTest]")
{
  // Loading data.
  data::DatasetInfo info;
  arma::mat trainData, testData;
  arma::rowvec trainResponses, testResponses;
  LoadBostonHousingDataset(trainData, testData, trainResponses, testResponses,
      info);

  // Add some noise.
  arma::mat noise(trainData.n_rows, 100, arma::fill::randu);
  arma::rowvec noiseResponses(100);
  for (size_t i = 0; i < noiseResponses.n_elem; ++i)
    noiseResponses[i] = 15 + Random(0, 10); // Random response.

  // Concatenate data matrices.
  arma::mat data = arma::join_rows(trainData, noise);
  arma::rowvec fullResponses = arma::join_rows(trainResponses, noiseResponses);

  // Now set weights.
  arma::rowvec weights(trainData.n_cols + 100);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = trainData.n_cols; i < trainData.n_cols + 100; ++i)
    weights[i] = Random(0.0, 0.01); // Low weights for false points.

  // Now build the decision tree.
  DecisionTreeRegressor<> d(data, fullResponses, weights);

  // Now we can check that we get good performance on the test set.
  arma::rowvec predictions;
  d.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the rmse.
  double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 6.0);
}

/**
 * Test that we can build a decision tree using weighted data (where the
 * low-weighted data is random noise) with MAD gain, and that the tree
 * still builds correctly enough to get good results.
 */
TEST_CASE("WeightedDecisionTreeMADGainTest", "[DecisionTreeRegressorTest]")
{
  // Loading data.
  data::DatasetInfo info;
  arma::mat trainData, testData;
  arma::rowvec trainResponses, testResponses;
  LoadBostonHousingDataset(trainData, testData, trainResponses, testResponses,
      info);

  // Add some noise.
  arma::mat noise(trainData.n_rows, 100, arma::fill::randu);
  arma::rowvec noiseResponses(100);
  for (size_t i = 0; i < noiseResponses.n_elem; ++i)
    noiseResponses[i] = 15 + Random(0, 10); // Random response.

  // Concatenate data matrices.
  arma::mat data = arma::join_rows(trainData, noise);
  arma::rowvec fullResponses = arma::join_rows(trainResponses, noiseResponses);

  // Now set weights.
  arma::rowvec weights(trainData.n_cols + 100);
  for (size_t i = 0; i < trainData.n_cols; ++i)
    weights[i] = Random(0.9, 1.0);
  for (size_t i = trainData.n_cols; i < trainData.n_cols + 100; ++i)
    weights[i] = Random(0.0, 0.01); // Low weights for false points.

  // Now build the decision tree using MADGain.
  DecisionTreeRegressor<MADGain> d(data, fullResponses, weights);

  // Now we can check that we get good performance on the test set.
  arma::rowvec predictions;
  d.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out the rmse.
  double rmse = RMSE(predictions, testResponses);
  REQUIRE(rmse < 6.5);
}
