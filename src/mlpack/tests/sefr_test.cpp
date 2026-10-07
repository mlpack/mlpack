/**
 * @file tests/sefr_test.cpp
 * @author Hamidreza Keshavarz
 *
 * Tests for the SEFR classifier.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/sefr.hpp>
#include <mlpack/methods/adaboost.hpp>

#include "catch.hpp"
#include "serialization.hpp"

using namespace mlpack;

TEST_CASE("SEFRSimpleBinaryTest", "[SEFRTest][tiny]")
{
  arma::mat data("0.1 0.2 0.3 0.9 0.8;"
                 "0.9 0.8 0.7 0.2 0.1");
  arma::Row<size_t> labels("0 0 0 1 1");

  SEFR<> sefr(data, labels, 2);

  REQUIRE(sefr.NumClasses() == 2);
  REQUIRE(sefr.Dimensionality() == 2);

  REQUIRE(sefr.Weights()(0, 1) == Approx(0.619047560091).epsilon(1e-8));
  REQUIRE(sefr.Weights()(1, 1) == Approx(-0.684210454294).epsilon(1e-8));
  REQUIRE(sefr.Biases()(1) == Approx(-0.084711774193).epsilon(1e-8));

  REQUIRE(sefr.Weights()(0, 0) == Approx(-sefr.Weights()(0, 1)));
  REQUIRE(sefr.Weights()(1, 0) == Approx(-sefr.Weights()(1, 1)));
  REQUIRE(sefr.Biases()(0) == Approx(-sefr.Biases()(1)));

  arma::Row<size_t> predictions;
  sefr.Classify(data, predictions);
  REQUIRE(arma::all(predictions == labels));

  arma::mat test("0.5 0.6 0.45;"
                 "0.5 0.4 0.55");
  arma::mat scores;
  sefr.Classify(test, predictions, scores);
  REQUIRE(predictions[0] == 0);
  REQUIRE(predictions[1] == 1);
  REQUIRE(predictions[2] == 0);
  REQUIRE(scores.n_rows == 2);
  REQUIRE(scores.n_cols == 3);
  REQUIRE(scores(1, 0) == Approx(-0.117293221295).epsilon(1e-8));
  REQUIRE(scores(1, 1) == Approx(0.013032580144).epsilon(1e-8));
  REQUIRE(scores(1, 2) == Approx(-0.182456122014).epsilon(1e-8));
}

TEST_CASE("SEFRClassifySinglePointTest", "[SEFRTest]")
{
  arma::mat data("0.1 0.2 0.3 0.9 0.8;"
                 "0.9 0.8 0.7 0.2 0.1");
  arma::Row<size_t> labels("0 0 0 1 1");

  SEFR<> sefr(data, labels, 2);

  arma::Row<size_t> predictions;
  arma::mat scores;
  sefr.Classify(data, predictions, scores);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t label;
    arma::vec pointScores;
    sefr.Classify(data.col(i), label, pointScores);

    REQUIRE(label == predictions[i]);
    REQUIRE(sefr.Classify(data.col(i)) == predictions[i]);
    REQUIRE(arma::approx_equal(pointScores, scores.col(i), "absdiff", 1e-10));
  }
}

TEST_CASE("SEFRIrisTest", "[SEFRTest]")
{
  arma::mat trainData, testData;
  arma::Mat<size_t> trainLabels, testLabels;
  if (!Load("iris_train.csv", trainData))
    FAIL("Cannot load dataset iris_train.csv");
  if (!Load("iris_train_labels.csv", trainLabels))
    FAIL("Cannot load labels for iris_train_labels.csv");
  if (!Load("iris_test.csv", testData))
    FAIL("Cannot load dataset iris_test.csv");
  if (!Load("iris_test_labels.csv", testLabels))
    FAIL("Cannot load labels for iris_test_labels.csv");

  SEFR<> sefr(trainData, trainLabels.row(0), 3);

  arma::Row<size_t> predictions;
  sefr.Classify(testData, predictions);

  const double accuracy = arma::accu(predictions == testLabels.row(0)) /
      (double) testLabels.n_elem;
  REQUIRE(accuracy >= 0.9);
}

TEMPLATE_TEST_CASE("SEFRIncrementalTest", "[SEFRTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;

  MatType data(5, 200, arma::fill::randu);
  arma::Row<size_t> labels(200);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) + data(1, i) > 1.0) ? (data(2, i) > 0.5 ? 2 : 1)
        : 0;

  SEFR<MatType> batch(data, labels, 3);
  SEFR<MatType> incremental(3, 5);
  for (size_t i = 0; i < data.n_cols; ++i)
    incremental.Train(data.col(i), labels[i]);

  const ElemType tol = std::is_same_v<ElemType, float> ? 1e-4 : 1e-10;
  REQUIRE(arma::approx_equal(batch.Weights(), incremental.Weights(), "absdiff",
      tol));
  REQUIRE(arma::approx_equal(batch.Biases(), incremental.Biases(), "absdiff",
      tol));
  REQUIRE(arma::approx_equal(batch.ClassCounts(), incremental.ClassCounts(),
      "absdiff", tol));
}

TEST_CASE("SEFRInstanceWeightsTest", "[SEFRTest]")
{
  arma::mat data(4, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) > 0.5) ? 1 : 0;

  // Integer weights should be equivalent to duplicating points.
  arma::rowvec weights(100);
  arma::mat duplicated;
  arma::Row<size_t> duplicatedLabels;
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    const size_t copies = 1 + (i % 3);
    weights[i] = copies;
    for (size_t j = 0; j < copies; ++j)
    {
      duplicated.insert_cols(duplicated.n_cols, data.col(i));
      duplicatedLabels.insert_cols(duplicatedLabels.n_cols,
          arma::Row<size_t>({ labels[i] }));
    }
  }

  SEFR<> weighted(data, labels, 2, weights);
  SEFR<> unweighted(duplicated, duplicatedLabels, 2);

  REQUIRE(arma::approx_equal(weighted.Weights(), unweighted.Weights(),
      "absdiff", 1e-10));
  REQUIRE(arma::approx_equal(weighted.Biases(), unweighted.Biases(),
      "absdiff", 1e-10));

  // Uniform weights should be equivalent to no weights.
  arma::rowvec uniformWeights(100);
  uniformWeights.fill(0.3);
  SEFR<> uniform(data, labels, 2, uniformWeights);
  SEFR<> plain(data, labels, 2);
  REQUIRE(arma::approx_equal(uniform.Weights(), plain.Weights(), "absdiff",
      1e-10));
  REQUIRE(arma::approx_equal(uniform.Biases(), plain.Biases(), "absdiff",
      1e-10));
}

TEST_CASE("SEFRSparseTest", "[SEFRTest]")
{
  arma::sp_mat sparseData;
  sparseData.sprandu(50, 300, 0.1);
  const arma::mat denseData(sparseData);
  arma::Row<size_t> labels(300);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (denseData(0, i) + denseData(1, i) > 0.0) ? 1 : 0;

  SEFR<> sparseModel(sparseData, labels, 2);
  SEFR<> denseModel(denseData, labels, 2);

  REQUIRE(arma::approx_equal(sparseModel.Weights(), denseModel.Weights(),
      "absdiff", 1e-10));
  REQUIRE(arma::approx_equal(sparseModel.Biases(), denseModel.Biases(),
      "absdiff", 1e-10));

  arma::Row<size_t> sparsePredictions, densePredictions;
  sparseModel.Classify(sparseData, sparsePredictions);
  denseModel.Classify(denseData, densePredictions);
  REQUIRE(arma::all(sparsePredictions == densePredictions));
}

TEST_CASE("SEFRMissingClassTest", "[SEFRTest]")
{
  arma::mat data(3, 50, arma::fill::randu);
  arma::Row<size_t> labels(50);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) > 0.5) ? 2 : 0;

  // Class 1 never appears, so it should never be predicted.
  SEFR<> sefr(data, labels, 3);
  arma::Row<size_t> predictions;
  sefr.Classify(arma::mat(3, 100, arma::fill::randu), predictions);
  REQUIRE(arma::all(predictions != 1));

  // With only one class seen, that class should always be predicted.
  SEFR<> incremental(3, 3);
  incremental.Train(data.col(0), 2);
  incremental.Classify(data, predictions);
  REQUIRE(arma::all(predictions == 2));
}

TEST_CASE("SEFREmptyDataTest", "[SEFRTest]")
{
  // Training on no points gives a model that can still be updated.
  SEFR<> sefr(arma::mat(3, 0), arma::Row<size_t>(), 2);
  REQUIRE(sefr.NumClasses() == 2);
  REQUIRE(sefr.Dimensionality() == 3);
  REQUIRE(arma::accu(sefr.ClassCounts()) == 0.0);

  sefr.Train(arma::vec("0.1 0.2 0.3"), 1);
  REQUIRE(sefr.Classify(arma::vec("0.5 0.5 0.5")) == 1);
}

TEST_CASE("SEFRResetTest", "[SEFRTest]")
{
  arma::mat data(3, 50, arma::fill::randu);
  arma::Row<size_t> labels(50);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(1, i) > 0.5) ? 1 : 0;

  SEFR<> sefr(data, labels, 2);
  sefr.Reset();

  REQUIRE(sefr.NumClasses() == 2);
  REQUIRE(sefr.Dimensionality() == 3);
  REQUIRE(arma::accu(arma::abs(sefr.Weights())) == 0.0);
  REQUIRE(arma::accu(arma::abs(sefr.ClassCounts())) == 0.0);

  // Training again after a reset should match a freshly trained model.
  for (size_t i = 0; i < data.n_cols; ++i)
    sefr.Train(data.col(i), labels[i]);

  SEFR<> fresh(data, labels, 2);
  REQUIRE(arma::approx_equal(sefr.Weights(), fresh.Weights(), "absdiff",
      1e-10));
  REQUIRE(arma::approx_equal(sefr.Biases(), fresh.Biases(), "absdiff", 1e-10));
}

TEST_CASE("SEFRRetrainTest", "[SEFRTest]")
{
  arma::mat data1(3, 50, arma::fill::randu), data2(4, 60, arma::fill::randu);
  arma::Row<size_t> labels1(50), labels2(60);
  for (size_t i = 0; i < labels1.n_elem; ++i)
    labels1[i] = (data1(0, i) > 0.5) ? 1 : 0;
  for (size_t i = 0; i < labels2.n_elem; ++i)
    labels2[i] = (data2(0, i) > 0.3) ? ((data2(1, i) > 0.5) ? 2 : 1) : 0;

  // Batch training discards any previous model.
  SEFR<> sefr(data1, labels1, 2);
  sefr.Train(data2, labels2, 3);
  SEFR<> fresh(data2, labels2, 3);

  REQUIRE(sefr.NumClasses() == 3);
  REQUIRE(sefr.Dimensionality() == 4);
  REQUIRE(arma::approx_equal(sefr.Weights(), fresh.Weights(), "absdiff",
      1e-10));
  REQUIRE(arma::approx_equal(sefr.Biases(), fresh.Biases(), "absdiff", 1e-10));
}

TEST_CASE("SEFRInvalidInputTest", "[SEFRTest]")
{
  arma::mat data(3, 10, arma::fill::randu);
  arma::Row<size_t> labels("0 1 0 1 0 1 0 1 0 1");

  REQUIRE_THROWS_AS(SEFR<>(data, labels.cols(0, 8), 2), std::invalid_argument);
  REQUIRE_THROWS_AS(SEFR<>(data, labels, 1), std::invalid_argument);
  REQUIRE_THROWS_AS(SEFR<>(data, labels + 1, 2), std::invalid_argument);
  REQUIRE_THROWS_AS(SEFR<>(data, labels, 2, arma::rowvec(5)),
      std::invalid_argument);

  SEFR<> sefr(data, labels, 2);
  arma::Row<size_t> predictions;
  REQUIRE_THROWS_AS(sefr.Classify(arma::mat(4, 5, arma::fill::randu),
      predictions), std::invalid_argument);
  REQUIRE_THROWS_AS(sefr.Classify(arma::vec(4, arma::fill::randu)),
      std::invalid_argument);
  REQUIRE_THROWS_AS(sefr.Train(arma::vec(4, arma::fill::randu), 0),
      std::invalid_argument);
  REQUIRE_THROWS_AS(sefr.Train(arma::vec(3, arma::fill::randu), 2),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE("SEFRModelMatTypeTest", "[SEFRTest]", float, double)
{
  using ElemType = TestType;
  using MatType = arma::Mat<ElemType>;

  MatType data(6, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) > 0.5) ? ((data(1, i) > 0.5) ? 3 : 2) :
        ((data(1, i) > 0.5) ? 1 : 0);

  SEFR<MatType> sefr(data, labels, 4);

  arma::Row<size_t> predictions;
  arma::Mat<ElemType> scores;
  sefr.Classify(data, predictions, scores);

  REQUIRE(scores.n_rows == 4);
  REQUIRE(scores.n_cols == 1000);
  const double accuracy = arma::accu(predictions == labels) /
      (double) labels.n_elem;
  REQUIRE(accuracy >= 0.75);
}

TEST_CASE("SEFRSerializationTest", "[SEFRTest]")
{
  arma::mat data(4, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(2, i) > 0.5) ? 1 : 0;

  SEFR<> sefr(data, labels, 2);
  SEFR<> xmlSefr, jsonSefr, binarySefr(3, 7);
  SerializeObjectAll(sefr, xmlSefr, jsonSefr, binarySefr);

  CheckMatrices(sefr.Weights(), xmlSefr.Weights(), jsonSefr.Weights(),
      binarySefr.Weights());
  CheckMatrices(sefr.Biases(), xmlSefr.Biases(), jsonSefr.Biases(),
      binarySefr.Biases());
  CheckMatrices(sefr.ClassSums(), xmlSefr.ClassSums(), jsonSefr.ClassSums(),
      binarySefr.ClassSums());
  CheckMatrices(sefr.ClassCounts(), xmlSefr.ClassCounts(),
      jsonSefr.ClassCounts(), binarySefr.ClassCounts());

  // A deserialized model can still be trained incrementally.
  arma::vec point(4, arma::fill::randu);
  sefr.Train(point, 1);
  binarySefr.Train(point, 1);
  REQUIRE(arma::approx_equal(sefr.Weights(), binarySefr.Weights(), "absdiff",
      1e-10));
}

TEST_CASE("SEFRAdaBoostTest", "[SEFRTest]")
{
  arma::mat trainData;
  arma::Mat<size_t> labels;
  if (!Load("iris_train.csv", trainData))
    FAIL("Cannot load dataset iris_train.csv");
  if (!Load("iris_train_labels.csv", labels))
    FAIL("Cannot load labels for iris_train_labels.csv");

  SEFR<> sefr(trainData, labels.row(0), 3);
  arma::Row<size_t> predictions;
  sefr.Classify(trainData, predictions);
  const size_t sefrErrors = arma::accu(predictions != labels.row(0));

  AdaBoost<SEFR<>> a(trainData, labels.row(0), 3, 25, 1e-10);
  a.Classify(trainData, predictions);
  const size_t adaBoostErrors = arma::accu(predictions != labels.row(0));

  REQUIRE(adaBoostErrors <= sefrErrors);
}

TEST_CASE("SEFRKFoldCVTest", "[SEFRTest]")
{
  arma::mat data("0 0 0 0 0 1 1 1 1 1");
  arma::Row<size_t> labels("0 0 0 0 0 1 1 1 1 1");
  const size_t numClasses = 2;

  KFoldCV<SEFR<>, Accuracy> cv(10, data, labels, numClasses);

  REQUIRE(cv.Evaluate() == Approx(1.0).epsilon(1e-7));
  REQUIRE_NOTHROW(cv.Model());
}

TEST_CASE("SEFRNonNegativeDataHasNoOffsetTest", "[SEFRTest]")
{
  arma::mat data(4, 100, arma::fill::randu);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) > 0.5) ? 1 : 0;

  SEFR<> sefr(data, labels, 2);

  REQUIRE(arma::all(sefr.Offsets() == 0.0));
}

TEST_CASE("SEFROffsetsAreClampedMinimumsTest", "[SEFRTest]")
{
  arma::mat data("-2.0  1.0  0.5;"
                 " 3.0  4.0  5.0;"
                 " 0.0 -1.5 -0.5");
  arma::Row<size_t> labels("0 1 1");

  SEFR<> sefr(data, labels, 2);

  REQUIRE(sefr.Offsets()[0] == Approx(-2.0));
  REQUIRE(sefr.Offsets()[1] == 0.0);
  REQUIRE(sefr.Offsets()[2] == Approx(-1.5));
}

TEST_CASE("SEFRShiftInvarianceTest", "[SEFRTest]")
{
  // Every feature has a minimum of exactly 0, so shifting the data down by 3
  // makes it negative and the offsets move it back to the same place: the
  // model must be the same up to the bias, and predictions must not change.
  arma::mat data(5, 300, arma::fill::randu);
  data.col(0).zeros();
  arma::Row<size_t> labels(300);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) + data(1, i) > 1.0) ? (data(2, i) > 0.5 ? 2 : 1)
        : 0;
  arma::mat test(5, 100, arma::fill::randu);

  const arma::mat shiftedData = data - 3.0;
  const arma::mat shiftedTest = test - 3.0;

  SEFR<> original(data, labels, 3);
  SEFR<> shifted(shiftedData, labels, 3);

  arma::vec expectedOffsets(5);
  expectedOffsets.fill(-3.0);
  REQUIRE(arma::approx_equal(shifted.Offsets(), expectedOffsets, "absdiff",
      1e-12));
  REQUIRE(arma::approx_equal(original.Weights(), shifted.Weights(), "absdiff",
      1e-10));

  arma::Row<size_t> originalPredictions, shiftedPredictions;
  arma::mat originalScores, shiftedScores;
  original.Classify(test, originalPredictions, originalScores);
  shifted.Classify(shiftedTest, shiftedPredictions, shiftedScores);
  REQUIRE(arma::all(originalPredictions == shiftedPredictions));
  REQUIRE(arma::approx_equal(originalScores, shiftedScores, "absdiff", 1e-10));
}

TEST_CASE("SEFRNegativeDataAccuracyTest", "[SEFRTest]")
{
  // Two Gaussian blobs centered at -2 and +2: separable, and impossible to use
  // without the offsets because the features are negative.
  arma::mat data(3, 400, arma::fill::randn);
  arma::Row<size_t> labels(400);
  for (size_t i = 0; i < labels.n_elem; ++i)
  {
    labels[i] = i % 2;
    data.col(i) += (labels[i] == 1) ? 2.0 : -2.0;
  }

  SEFR<> sefr(data, labels, 2);
  arma::Row<size_t> predictions;
  sefr.Classify(data, predictions);

  REQUIRE(arma::all(sefr.Offsets() < 0.0));
  const double accuracy = arma::accu(predictions == labels) /
      (double) labels.n_elem;
  REQUIRE(accuracy >= 0.95);
}

TEMPLATE_TEST_CASE("SEFRNegativeDataIncrementalTest", "[SEFRTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;

  MatType data(4, 200, arma::fill::randn);
  arma::Row<size_t> labels(200);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(0, i) > 0.0) ? 1 : 0;

  SEFR<MatType> batch(data, labels, 2);
  SEFR<MatType> incremental(2, 4);
  for (size_t i = 0; i < data.n_cols; ++i)
    incremental.Train(data.col(i), labels[i]);

  const ElemType tol = std::is_same_v<ElemType, float> ? 1e-3 : 1e-10;
  REQUIRE(arma::approx_equal(batch.Offsets(), incremental.Offsets(),
      "absdiff", tol));
  REQUIRE(arma::approx_equal(batch.Weights(), incremental.Weights(), "absdiff",
      tol));
  REQUIRE(arma::approx_equal(batch.Biases(), incremental.Biases(), "absdiff",
      tol));
}

TEST_CASE("SEFRNegativeSparseTest", "[SEFRTest]")
{
  arma::sp_mat sparseData;
  sparseData.sprandn(30, 200, 0.1);
  const arma::mat denseData(sparseData);
  arma::Row<size_t> labels(200);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (denseData(0, i) + denseData(1, i) > 0.0) ? 1 : 0;

  SEFR<> sparseModel(sparseData, labels, 2);
  SEFR<> denseModel(denseData, labels, 2);

  REQUIRE(arma::approx_equal(sparseModel.Offsets(), denseModel.Offsets(),
      "absdiff", 1e-12));
  REQUIRE(arma::approx_equal(sparseModel.Weights(), denseModel.Weights(),
      "absdiff", 1e-10));
  REQUIRE(arma::approx_equal(sparseModel.Biases(), denseModel.Biases(),
      "absdiff", 1e-10));
}

TEST_CASE("SEFRNegativeDataSerializationTest", "[SEFRTest]")
{
  arma::mat data(3, 100, arma::fill::randn);
  arma::Row<size_t> labels(100);
  for (size_t i = 0; i < labels.n_elem; ++i)
    labels[i] = (data(1, i) > 0.0) ? 1 : 0;

  SEFR<> sefr(data, labels, 2);
  SEFR<> xmlSefr, jsonSefr, binarySefr;
  SerializeObjectAll(sefr, xmlSefr, jsonSefr, binarySefr);

  CheckMatrices(sefr.Offsets(), xmlSefr.Offsets(), jsonSefr.Offsets(),
      binarySefr.Offsets());

  // A point below the training minimum moves the offsets after loading too.
  const arma::vec point("-10.0 -10.0 -10.0");
  sefr.Train(point, 0);
  binarySefr.Train(point, 0);
  REQUIRE(arma::approx_equal(sefr.Offsets(), binarySefr.Offsets(), "absdiff",
      1e-12));
  REQUIRE(arma::approx_equal(sefr.Weights(), binarySefr.Weights(), "absdiff",
      1e-10));
}
