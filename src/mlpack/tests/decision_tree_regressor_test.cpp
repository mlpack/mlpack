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
#include <mlpack/methods/decision_tree/decision_tree_regressor.hpp>
#include <mlpack/methods/decision_tree/mad_gain.hpp>
#include <mlpack/methods/decision_tree/mse_gain.hpp>
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/methods/decision_tree/multiple_random_dimension_select.hpp>
#include <mlpack/core/math/random.hpp>

#include "catch.hpp"
#include "serialization.hpp"
#include "mock_categorical_data.hpp"
#include "test_function_tools.hpp"

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::distribution;

/**
 * Creates dataset with 5 groups with all the points in same group have exactly
 * same label.
 */
void CreateMultiSplitData(arma::mat& d, arma::rowvec& l, const size_t count,
    arma::rowvec& values)
{
  d = arma::mat(10, count, arma::fill::randu);
  l = arma::rowvec(count);

  // Group 1.
  for (size_t i = 0; i < count / 5; i++)
  {
    d(3, i) = i;
    l(i) = values[0];
  }
  // Group 2.
  for (size_t i = count / 5; i < (count / 5) * 2; i++)
  {
    d(3, i) = i;
    l(i) = values[1];
  }
  // Group 3.
  for (size_t i = (count / 5) * 2; i < (count / 5) * 3; i++)
  {
    d(3, i) = i;
    l(i) = values[2];
  }
  // Group 4.
  for (size_t i = (count / 5) * 3; i < (count / 5) * 4; i++)
  {
    d(3, i) = i;
    l(i) = values[3];
  }
  // Group 5.
  for (size_t i = (count / 5) * 4; i < count; i++)
  {
    d(3, i) = i;
    l(i) = values[4];
  }
}

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
 * Making sure that MSE gain is evaluated correctly by doing calculation by
 * hand.
 */
TEST_CASE("MSEGainHandCalculation", "[DecisionTreeRegressorTest]")
{
  arma::rowvec labels = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  arma::rowvec weights = {0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7};

  // Hand calculated gain values.
  const double gain = -27.08999;
  const double weightedGain = -27.53960;
  REQUIRE(MSEGain::Evaluate<false>(labels, 0, weights) ==
          Approx(gain).margin(1e-5));
  REQUIRE(MSEGain::Evaluate<true>(labels, 0, weights) ==
          Approx(weightedGain).margin(1e-5));
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
 * Making sure that MAD gain is evaluated correctly by doing calculation by
 * hand.
 */
TEST_CASE("MADGainHandCalculation", "[DecisionTreeRegressorTest]")
{
  arma::rowvec labels = {4., 2., 3., 4., 13., 6., 20., 8., 9., 10.};
  arma::rowvec weights = {0.3, 0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7};

  // Hand calculated gain values.
  const double gain = -4.1;
  const double weightedGain = -3.8592;
  REQUIRE(MADGain::Evaluate<false>(labels, 0, weights) ==
          Approx(gain).margin(1e-5));
  REQUIRE(MADGain::Evaluate<true>(labels, 0, weights) ==
          Approx(weightedGain).margin(1e-5));
}

/**
 * Check that AllCategoricalSplit will split when the split is obviously
 * better.
 */
TEST_CASE("AllCategoricalSplitSimpleSplitTest_", "[DecisionTreeRegressorTest]")
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
TEST_CASE("AllCategoricalSplitMinSamplesTest_", "[DecisionTreeRegressorTest]")
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
TEST_CASE("AllCategoricalSplitNoGainTest_", "[DecisionTreeRegressorTest]")
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
TEST_CASE("BestBinaryNumericSplitSimpleSplitTest_", "[DecisionTreeRegressorTest]")
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
      bestGain, predictors, labels, weights, 3, 1e-7, splitInfo, aux);
  const double weightedGain =
      BestBinaryNumericSplit<MADGain>::SplitIfBetter<true>(bestGain, predictors,
      labels, weights, 3, 1e-7, splitInfo, aux);

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
TEST_CASE("BestBinaryNumericSplitMinSamplesTest_", "[DecisionTreeRegressorTest]")
{
  arma::rowvec predictors = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
  arma::rowvec labels = { 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  arma::rowvec weights(labels.n_elem);

  double splitInfo;
  BestBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 0, weights);
  const double gain = BestBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, predictors, labels, weights, 8, 1e-7, splitInfo, aux);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      BestBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, predictors,
      labels, weights, 8, 1e-7, splitInfo, aux);

  // Make sure that no split was made.
  REQUIRE(gain == DBL_MAX);
  REQUIRE(gain == weightedGain);
}

/**
 * Check that the BestBinaryNumericSplit doesn't split a dimension that gives no
 * gain.
 */
TEST_CASE("BestBinaryNumericSplitNoGainTest_", "[DecisionTreeRegressorTest]")
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
      bestGain, predictors, labels, weights, 10, 1e-7, splitInfo,
      aux);

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
  arma::rowvec labels("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights;
  weights.ones(labels.n_elem);

  double splitInfo;
  RandomBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 2, weights);
  const double gain = RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, labels, weights, 1, 1e-7, splitInfo, aux);
  const double weightedGain =
      RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, values,
      labels, weights, 1, 1e-7, splitInfo, aux);

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
  arma::rowvec labels("0 0 0 0 0 1 1 1 1 1 1");
  arma::rowvec weights(labels.n_elem);

  double splitInfo;
  RandomBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 2, weights);
  const double gain = RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, labels, weights, 8, 1e-7, splitInfo, aux);
  // This should make no difference because it won't split at all.
  const double weightedGain =
      RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<true>(bestGain, values,
      labels, weights, 8, 1e-7, splitInfo, aux);

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
  arma::Row<double> labels(100);
  arma::rowvec weights;
  for (size_t i = 0; i < 100; i += 2)
  {
    values[i] = i;
    labels[i] = 0.0;
    values[i + 1] = i;
    labels[i + 1] = 1.0;
  }

  double splitInfo;
  RandomBinaryNumericSplit<MSEGain>::AuxiliarySplitInfo aux;

  // Call the method to do the splitting.
  const double bestGain = MSEGain::Evaluate<false>(labels, 2, weights);
  const double gain = RandomBinaryNumericSplit<MSEGain>::SplitIfBetter<false>(
      bestGain, values, labels, weights, 10, 1e-7, splitInfo, aux, true);

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
  arma::Row<double> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 1.0;
  }

  // Use default parameters.
  DecisionTreeRegressor<> d(dataset, labels);

  // Now require that we have some children.
  REQUIRE(d.NumChildren() > 0);
}

/**
 * Construct a tree with weighted labels.
 */
TEST_CASE("BasicConstructionTestWithWeight_", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::Row<double> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 1.0;
  }
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Use default parameters.
  DecisionTreeRegressor<> wd(dataset, labels, weights);
  DecisionTreeRegressor<> d(dataset, labels);

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
  arma::Row<double> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 1.0;
  }

  DecisionTreeRegressor<> d(dataset, labels, 1, 0.0); // Minimum leaf size of 1.

  // Make sure that we can get perfect accuracy on the training set.
  for (size_t i = 0; i < 100; ++i)
  {
    double prediction;
    prediction = d.Predict(dataset.col(i));

    REQUIRE(prediction == Approx(labels[i]).epsilon(1e-7));
  }
}

/**
 * Construct the decision tree with weighted labels
 */
TEST_CASE("PerfectTrainingSetWithWeight_", "[DecisionTreeRegressorTest]")
{
  // Completely random dataset with no structure.
  arma::mat dataset(10, 100, arma::fill::randu);
  arma::Row<double> labels(100);
  for (size_t i = 0; i < 50; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 0.0;
  }
  for (size_t i = 50; i < 100; ++i)
  {
    dataset(3, i) = i;
    labels[i] = 1.0;
  }
  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, labels, weights, 1, 0.0);

  // This part of code is dupliacte with no weighted one.
  for (size_t i = 0; i < 100; ++i)
  {
    size_t prediction;
    prediction = d.Predict(dataset.col(i));

    REQUIRE(prediction == Approx(labels[i]).epsilon(1e-7));
  }
}

/**
 * Test that we can build a decision tree on a simple categorical dataset.
 */
TEST_CASE("CategoricalBuildTest_", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingLabels = l.subvec(0, 1999);
  arma::rowvec testLabels = l.subvec(2000, 3999);

  // Build the tree.
  DecisionTreeRegressor<> tree(trainingData, di, trainingLabels, 10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testLabels);
  REQUIRE(rmse < 1.0);
}

/**
 * Test that we can build a decision tree with weights on a simple categorical
 * dataset.
 */
TEST_CASE("CategoricalBuildTestWithWeight_", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingLabels = l.subvec(0, 1999);
  arma::rowvec testLabels = l.subvec(2000, 3999);

  arma::rowvec weights = arma::ones<arma::rowvec>(
      trainingLabels.n_elem);

  // Build the tree.
  DecisionTreeRegressor<> tree(trainingData, di, trainingLabels, weights, 10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testLabels);
  REQUIRE(rmse < 1.0);
}

/**
 * Test that we can build a decision tree using weighted data (where the
 * low-weighted data is random noise), and that the tree still builds correctly
 * enough to get good results.
 */
// TEST_CASE("WeightedDecisionTreeTest_", "[DecisionTreeRegressorTest]")
// {
//   // Loading data.
//   data::DatasetInfo info;
//   arma::mat trainData, testData;
//   arma::Row<double> trainLabels, testLabels;
//   LoadBostonHousingDataset(trainData, testData, trainLabels, testLabels, info);

//   // Add some noise.
//   arma::mat noise(trainData.n_rows, 500, arma::fill::randu);
//   arma::Row<double> noiseLabels(500);
//   for (size_t i = 0; i < noiseLabels.n_elem; ++i)
//     noiseLabels[i] = 15 + math::Random(0, 10); // Random label.

//   // Concatenate data matrices.
//   arma::mat data = arma::join_rows(trainData, noise);
//   arma::Row<double> fullLabels = arma::join_rows(trainLabels, noiseLabels);

//   // Now set weights.
//   arma::rowvec weights(trainData.n_cols + 500);
//   for (size_t i = 0; i < trainData.n_cols; ++i)
//     weights[i] = math::Random(0.9, 1.0);
//   for (size_t i = trainData.n_cols; i < trainData.n_cols + 500; ++i)
//     weights[i] = math::Random(0.0, 0.01); // Low weights for false points.

//   // Now build the decision tree.  I think the syntax is right here.
//   DecisionTreeRegressor<> d(data, fullLabels, weights);

//   // Now we can check that we get good performance on the VC2 test set.
//   arma::Row<double> predictions;
//   d.Predict(testData, predictions);

//   REQUIRE(predictions.n_elem == testData.n_cols);

//   // Figure out the accuracy.
//   double rmse = RMSE(predictions, testLabels);

//   REQUIRE(rmse < 9.21);
// }

/**
 * Test that we can build a decision tree on a simple categorical dataset using
 * weights, with low-weight noise added.
 */
TEST_CASE("CategoricalWeightedBuildTest_", "[DecisionTreeRegressorTest]")
{
  arma::mat d;
  arma::rowvec l;
  data::DatasetInfo di;
  MockCategoricalData(d, l, di);

  // Split into a training set and a test set.
  arma::mat trainingData = d.cols(0, 1999);
  arma::mat testData = d.cols(2000, 3999);
  arma::rowvec trainingLabels = l.subvec(0, 1999);
  arma::rowvec testLabels = l.subvec(2000, 3999);

  // Now create random points.
  arma::mat randomNoise(5, 2000);
  arma::rowvec randomLabels(2000);
  for (size_t i = 0; i < 2000; ++i)
  {
    randomNoise(0, i) = math::Random();
    randomNoise(1, i) = math::Random(-1, 1);
    randomNoise(2, i) = math::Random();
    randomNoise(3, i) = math::RandInt(0, 2);
    randomNoise(4, i) = math::RandInt(0, 5);
    randomLabels[i] = math::Random(-10, 18);
  }

  // Generate weights.
  arma::rowvec weights(4000);
  for (size_t i = 0; i < 2000; ++i)
    weights[i] = math::Random(0.9, 1.0);
  for (size_t i = 2000; i < 4000; ++i)
    weights[i] = math::Random(0.0, 0.001);

  arma::mat fullData = arma::join_rows(trainingData, randomNoise);
  arma::rowvec fullLabels = arma::join_rows(trainingLabels, randomLabels);

  // Build the tree.
  DecisionTreeRegressor<> tree(fullData, di, fullLabels, weights, 10);

  // Now evaluate the quality of predictions.
  arma::rowvec predictions;
  tree.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Make sure we get reasonable rmse.
  const double rmse = RMSE(predictions, testLabels);
  REQUIRE(rmse < 1.5);
}

// /**
//  * Test that we can build a decision tree using weighted data (where the
//  * low-weighted data is random noise) with information gain, and that the tree
//  * still builds correctly enough to get good results.
//  */
// TEST_CASE("WeightedDecisionTreeInformationGainTest_",
//     "[DecisionTreeRegressorTest]")
// {
//   arma::mat dataset;
//   arma::Row<double> labels;
//   if (!data::Load("vc2.csv", dataset))
//     FAIL("Cannot load test dataset vc2.csv!");
//   if (!data::Load("vc2_labels.txt", labels))
//     FAIL("Cannot load labels for vc2_labels.txt!");

//   // Add some noise.
//   arma::mat noise(dataset.n_rows, 1000, arma::fill::randu);
//   arma::Row<double> noiseLabels(1000);
//   for (size_t i = 0; i < noiseLabels.n_elem; ++i)
//     noiseLabels[i] = math::Random(0, 3); // Random label.

//   // Concatenate data matrices.
//   arma::mat data = arma::join_rows(dataset, noise);
//   arma::Row<double> fullLabels = arma::join_rows(labels, noiseLabels);

//   // Now set weights.
//   arma::rowvec weights(dataset.n_cols + 1000);
//   for (size_t i = 0; i < dataset.n_cols; ++i)
//     weights[i] = math::Random(0.9, 1.0);
//   for (size_t i = dataset.n_cols; i < dataset.n_cols + 1000; ++i)
//     weights[i] = math::Random(0.0, 0.01); // Low weights for false points.

//   // Now build the decision tree.  I think the syntax is right here.
//   DecisionTreeRegressor<MADGain> d(data, fullLabels, weights);

//   // Now we can check that we get good performance on the VC2 test set.
//   arma::mat testData;
//   arma::Row<double> testLabels;
//   if (!data::Load("vc2_test.csv", testData))
//     FAIL("Cannot load test dataset vc2_test.csv!");
//   if (!data::Load("vc2_test_labels.txt", testLabels))
//     FAIL("Cannot load labels for vc2_test_labels.txt!");

//   arma::Row<double> predictions;
//   d.Predict(testData, predictions);

//   REQUIRE(predictions.n_elem == testData.n_cols);

//   // Figure out the accuracy.
//   double accuracy = R2Score(predictions, testLabels);

//   REQUIRE(accuracy > 0.75);
// }

// /**
//  * Test that we can build a decision tree using information gain on a simple
//  * categorical dataset using weights, with low-weight noise added.
//  */
// TEST_CASE("CategoricalInformationGainWeightedBuildTest", "[DecisionTreeTest]")
// {
//   arma::mat d;
//   arma::Row<size_t> l;
//   data::DatasetInfo di;
//   MockCategoricalData(d, l, di);

//   // Split into a training set and a test set.
//   arma::mat trainingData = d.cols(0, 1999);
//   arma::mat testData = d.cols(2000, 3999);
//   arma::Row<size_t> trainingLabels = l.subvec(0, 1999);
//   arma::Row<size_t> testLabels = l.subvec(2000, 3999);

//   // Now create random points.
//   arma::mat randomNoise(4, 2000);
//   arma::Row<size_t> randomLabels(2000);
//   for (size_t i = 0; i < 2000; ++i)
//   {
//     randomNoise(0, i) = math::Random();
//     randomNoise(1, i) = math::Random();
//     randomNoise(2, i) = math::RandInt(4);
//     randomNoise(3, i) = math::RandInt(2);
//     randomLabels[i] = math::RandInt(5);
//   }

//   // Generate weights.
//   arma::rowvec weights(4000);
//   for (size_t i = 0; i < 2000; ++i)
//     weights[i] = math::Random(0.9, 1.0);
//   for (size_t i = 2000; i < 4000; ++i)
//     weights[i] = math::Random(0.0, 0.001);

//   arma::mat fullData = arma::join_rows(trainingData, randomNoise);
//   arma::Row<size_t> fullLabels = arma::join_rows(trainingLabels, randomLabels);

//   // Build the tree.
//   DecisionTree<InformationGain> tree(fullData, di, fullLabels, 5, weights, 10);

//   // Now evaluate the accuracy of the tree.
//   arma::Row<size_t> predictions;
//   tree.Classify(testData, predictions);

//   REQUIRE(predictions.n_elem == testData.n_cols);
//   size_t correct = 0;
//   for (size_t i = 0; i < testData.n_cols; ++i)
//     if (testLabels[i] == predictions[i])
//       ++correct;

//   // Make sure we got at least 70% accuracy.
//   const double correctPct = double(correct) / double(testData.n_cols);
//   REQUIRE(correctPct > 0.70);
// }

/**
 * Test that the decision tree generalizes reasonably.
 */
TEST_CASE("SimpleGeneralizationTest_", "[DecisionTreeRegressorTest]")
{

  // Loading data.
  data::DatasetInfo info;
  arma::mat trainData, testData;
  arma::Row<double> trainLabels, testLabels;
  arma::rowvec weights = arma::ones<arma::rowvec>(355);
  LoadBostonHousingDataset(trainData, testData, trainLabels, testLabels, info);

  // Build decision tree.
  DecisionTreeRegressor<MSEGain> d(trainData, info, trainLabels);

  // Get the predicted test labels.
  arma::Row<double> predictions;
  d.Predict(testData, predictions);

  REQUIRE(predictions.n_elem == testData.n_cols);

  // Figure out rmse.
  double rmse = RMSE(predictions, testLabels);

  // REQUIRE(rmse < 9.21);
  // std::cout << predictions << std::endl << testLabels;
  arma::Row<double> trainPred;
  d.Predict(trainData, trainPred);
  // std::cout << trainPred;

  std::cout << "Train RMSE: " << RMSE(trainLabels, trainPred) << std::endl;
}

/**
 * Test that the decision tree generalizes reasonably when built on float data.
 */
// TEST_CASE("SimpleGeneralizationFMatTest_", "[DecisionTreeRegressorTest]")
// {
//   // Loading data.
//   data::DatasetInfo info;
//   arma::mat trainData, testData;
//   arma::Row<double> trainLabels, testLabels;
//   LoadBostonHousingDataset(trainData, testData, trainLabels, testLabels, info);

//   // Initialize an all-ones weight matrix.
//   arma::rowvec weights(trainLabels.n_cols, arma::fill::ones);

//   // Build decision tree.
//   DecisionTreeRegressor<> d(trainData, trainLabels);
//   DecisionTreeRegressor<> wd(trainData, trainLabels, weights);

//   // Get the predicted test labels.
//   arma::Row<double> predictions;
//   d.Predict(testData, predictions);

//   REQUIRE(predictions.n_elem == testData.n_cols);

//   // Figure out the rmse.
//   double rmse = RMSE(predictions, testLabels);

//   REQUIRE(rmse < 9.21);
//   std::cout << R2Score(predictions, testLabels) << std::endl;

//   // Reset the prediction.
//   predictions.zeros();
//   wd.Predict(testData, predictions);

//   REQUIRE(predictions.n_elem == testData.n_cols);

//   // Figure out the rmse.
//   double wdrmse = RMSE(predictions, testLabels);

//   REQUIRE(wdrmse < 9.21);
// }

// TEST_CASE("DecisionTreeRegressorEnergyTest", "[DecisionTreeRegressorTest]")
// {
//   arma::mat m;
//   if (!data::Load("energydata_complete.csv", m))
//     FAIL("Cannot load dataset energydata_complete.csv!");

//   arma::rowvec r = m.row(0);
//   m.shed_row(0);

//   DecisionTreeRegressor<> d(m, r, 1, 0.0, 0);

//   arma::rowvec p;
//   d.Predict(m, p);

//   const double mse = arma::accu(arma::square(p - r)) / p.n_elem;
//   REQUIRE(mse == Approx(0.0).epsilon(1e-4));
// }

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
  arma::rowvec labels;
  arma::rowvec values = {0.0, 1.0, 2.0, 1.0, 0.0};

  CreateMultiSplitData(dataset, labels, 1000, values);

  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, labels, weights, 2, 0.0);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  for (size_t i = 0; i < labels.n_elem; ++i)
    REQUIRE(preds[i] == labels[i]);
}

/**
 * Test that the tree is able to perfectly fit all the obvious splits present
 * in the data. Same test as above, but with less data.
 */
TEST_CASE("MultiSplitTest2", "[DecisionTreeRegressorTest]")
{
  arma::mat dataset;
  arma::rowvec labels;
  arma::rowvec values = {0.0, 1.0, 2.0, 1.0, 0.0};

  CreateMultiSplitData(dataset, labels, 100, values);

  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, labels, weights, 2, 0.0);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  for (size_t i = 0; i < labels.n_elem; ++i)
    REQUIRE(preds[i] == labels[i]);
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
  arma::Row<double> labels;
  arma::rowvec values = {0.0, 5.0, 10.0, 15.0, 20.0};

  CreateMultiSplitData(dataset, labels, 500, values);

  arma::rowvec weights(labels.n_elem);
  weights.ones();

  // Minimum leaf size of 1.
  DecisionTreeRegressor<> d(dataset, labels, weights, 2, 0.0);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  for (size_t i = 0; i < labels.n_elem; ++i)
    REQUIRE(preds[i] == labels[i]);
}

TEST_CASE("handmadedata", "[DecisionTreeRegressorTest]")
{
  // drug dosage (in mg).
  arma::mat dataset = {{2, 3, 5, 10, 14, 16, 20, 22, 28, 30, 32, 35, 39}};
  // percentage effectiveness.
  arma::rowvec labels = {0, 0, 0, 5, 99, 99, 99, 95, 55, 45, 7, 0, 0};

  arma::rowvec weights(labels.n_elem);
  weights.ones();
  std::cout << "****************Start**************\n";
  DecisionTreeRegressor<> d(dataset, labels, weights, 2, 0.0, 5);
  arma::rowvec preds;
  d.Predict(dataset, preds);

  const double mse = arma::accu(arma::square(preds - labels)) / preds.n_elem;
  REQUIRE(mse == Approx(0.0).epsilon(1e-4));
  std::cout << "****************End****************\n";
}
