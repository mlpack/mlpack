/**
 * @file tests/metric_test.cpp
 *
 * Unit tests for the various metrics.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include "catch.hpp"
#include <mlpack/core/metrics/iou_metric.hpp>
#include <mlpack/core/metrics/non_maximal_supression.hpp>
#include <mlpack/core/metrics/bleu.hpp>
#include "test_catch_tools.hpp"

using namespace std;
using namespace mlpack::metric;

/**
 * Simple test for L-1 metric.
 */
TEST_CASE("L1MetricTest", "[MetricTest]")
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::Col<size_t> b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  ManhattanDistance lMetric;

  REQUIRE((double) arma::accu(arma::abs(a1 - b1)) ==
      Approx(lMetric.Evaluate(a1, b1)).epsilon(1e-7));

  REQUIRE((double) arma::accu(arma::abs(a2 - b2)) ==
      Approx(lMetric.Evaluate(a2, b2)).epsilon(1e-7));
}

/**
 * Simple test for L-2 metric.
 */
TEST_CASE("L2MetricTest", "[MetricTest]")
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::vec a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::vec b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  EuclideanDistance lMetric;

  REQUIRE((double) sqrt(arma::accu(arma::square(a1 - b1))) ==
      Approx(lMetric.Evaluate(a1, b1)).epsilon(1e-7));

  REQUIRE((double) sqrt(arma::accu(arma::square(a2 - b2))) ==
      Approx(lMetric.Evaluate(a2, b2)).epsilon(1e-7));
}

/**
 * Simple test for L-Infinity metric.
 */
TEST_CASE("LINFMetricTest", "[MetricTest]")
{
  arma::vec a1(5);
  a1.randn();

  arma::vec b1(5);
  b1.randn();

  arma::Col<size_t> a2(5);
  a2 << 1 << 2 << 1 << 0 << 5;

  arma::Col<size_t> b2(5);
  b2 << 2 << 5 << 2 << 0 << 1;

  ChebyshevDistance lMetric;

  REQUIRE((double) arma::as_scalar(arma::max(arma::abs(a1 - b1))) ==
      Approx(lMetric.Evaluate(a1, b1)).epsilon(1e-7));

  REQUIRE((double) arma::as_scalar(arma::max(arma::abs(a2 - b2))) ==
      Approx(lMetric.Evaluate(a2, b2)).epsilon(1e-7));
}

/**
 * Simple test for IoU metric.
 */
TEST_CASE("IoUMetricTest", "[MetricTest]")
{
  arma::vec bbox1(4), bbox2(4);
  bbox1 << 1 << 2 << 100 << 200;
  bbox2 << 1 << 2 << 100 << 200;
  // IoU of same bounding boxes equals 1.0.
  REQUIRE(1.0 == Approx(IoU<>::Evaluate(bbox1, bbox2)).epsilon(1e-6));

  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, x1, y1}.
  bbox1 << 39 << 63 << 203 << 112;
  bbox2 << 54 << 66 << 198 << 114;
  // Value calculated using Python interpreter.
  REQUIRE(IoU<true>::Evaluate(bbox1, bbox2) ==
      Approx(0.7980093).epsilon(1e-6));

  bbox1 << 31 << 69 << 201 << 125;
  bbox2 << 18 << 63 << 235 << 135;
  // Value calculated using Python interpreter.
  REQUIRE(IoU<true>::Evaluate(bbox1, bbox2) ==
      Approx(0.612479577).epsilon(1e-6));

  // Use hieght - width representation of bounding boxes.
  // Bounding boxes represent {x0, y0, h, w}.
  bbox1 << 49 << 75 << 154 << 50;
  bbox2 << 42 << 78 << 144 << 48;
  // Value calculated using Python interpreter.
  REQUIRE(IoU<>::Evaluate(bbox1, bbox2) == Approx(0.7898879).epsilon(1e-6));

  bbox1 << 35 << 51 << 161 << 59;
  bbox2 << 36 << 60 << 144 << 48;
  // Value calculated using Python interpreter.
  REQUIRE(IoU<>::Evaluate(bbox1, bbox2) == Approx(0.7309670).epsilon(1e-6));
}

TEST_CASE("NMSMetricTest", "[MetricTest]")
{
  arma::mat bbox, selectedBoundingBox, desiredBoundingBox;
  arma::vec bbox1(4), bbox2(4), bbox3(4);
  arma::uvec selectedIndices, desiredIndices;

  // Set values of each bounding box.
  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, x1, y1}.
  bbox1 << 0.5 << 0.5 << 41.0 << 31.0;
  bbox2 << 1.0 << 1.0 << 42.0 << 22.0;
  bbox3 << 10.0 << 13.0 << 90.0 << 100.0;

  // Fill bounding box.
  bbox.insert_cols(0, bbox3);
  bbox.insert_cols(0, bbox2);
  bbox.insert_cols(0, bbox1);

  // Fill confidence scores for each bounding box.
  arma::vec confidenceScores(3);
  confidenceScores << 0.7 << 0.6 << 0.4;

  // Selected bounding box using torchvision.ops.nms().
  desiredBoundingBox.insert_cols(0, bbox3);
  desiredBoundingBox.insert_cols(0, bbox1);

  // Selected indices of bounding boxes using
  // torchvision.ops.nms().
  desiredIndices = arma::ucolvec(2);
  desiredIndices << 0 << 2;

  // Evaluate the bounding box.
  NMS<true>::Evaluate(bbox, confidenceScores,
      selectedIndices);

  selectedBoundingBox = bbox.cols(selectedIndices);

  REQUIRE(selectedBoundingBox.n_cols == 2);
  REQUIRE(selectedBoundingBox.n_rows == 4);
  CheckMatrices(desiredBoundingBox, selectedBoundingBox);

  for (size_t i = 0; i < desiredIndices.n_elem; i++)
  {
    REQUIRE(desiredIndices[i] == selectedIndices[i]);
  }

  // Clean up.
  bbox.clear();
  desiredBoundingBox.clear();
  selectedBoundingBox.clear();

  // Fill new bounding boxes.
  bbox.insert_cols(0, bbox1);
  bbox.insert_cols(0, bbox2);
  bbox.insert_cols(0, bbox1);
  confidenceScores << 1.0 << 0.6 << 0.9;

  // Output calculated using using torchvision.ops.nms().
  desiredBoundingBox.insert_cols(0, bbox2);
  desiredBoundingBox.insert_cols(0, bbox1);

  NMS<true>::Evaluate(bbox, confidenceScores,
      selectedIndices, 0.9);

  selectedBoundingBox = bbox.cols(selectedIndices);

  REQUIRE(selectedBoundingBox.n_cols == 2);
  REQUIRE(selectedBoundingBox.n_rows == 4);
  CheckMatrices(desiredBoundingBox, selectedBoundingBox);

  // Clean up.
  bbox.clear();
  desiredBoundingBox.clear();
  selectedBoundingBox.clear();

  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, x1, y1}.
  bbox1 << 39 << 63 << 203 << 112;
  bbox2 << 31 << 69 << 201 << 125;
  bbox3 << 54 << 66 << 198 << 114;

  // Fill bounding box.
  bbox.insert_cols(0, bbox3);
  bbox.insert_cols(0, bbox2);
  bbox.insert_cols(0, bbox1);

  // Fill confidence scores of bounding boxes.
  confidenceScores << 1.0 << 0.6 << 0.9;

  // Selected bounding box using torchvision.ops.nms().
  desiredBoundingBox.insert_cols(0, bbox2);
  desiredBoundingBox.insert_cols(0, bbox1);

  NMS<true>::Evaluate(bbox, confidenceScores,
      selectedIndices, 0.7);

  selectedBoundingBox = bbox.cols(selectedIndices);

  REQUIRE(selectedBoundingBox.n_cols == 2);
  REQUIRE(selectedBoundingBox.n_rows == 4);
  CheckMatrices(desiredBoundingBox, selectedBoundingBox);

  // Clean up.
  bbox.clear();
  desiredBoundingBox.clear();
  selectedBoundingBox.clear();

  // Set values of each bounding box.
  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, h, w}.
  bbox1 << 0.0 << 0.0 << 41.0 << 31.0;
  bbox2 << 1.0 << 1.0 << 41.0 << 21.0;
  bbox3 << 10.0 << 13.0 << 80.0 << 87.0;

  // Fill bounding box.
  bbox.insert_cols(0, bbox3);
  bbox.insert_cols(0, bbox2);
  bbox.insert_cols(0, bbox1);

  // Fill confidence scores for each bounding box.
  confidenceScores << 0.7 << 0.6 << 0.4;

  // Selected bounding box using torchvision.ops.nms().
  desiredBoundingBox.insert_cols(0, bbox3);
  desiredBoundingBox.insert_cols(0, bbox1);

  // Evaluate the bounding box.
  NMS<>::Evaluate(bbox, confidenceScores,
    selectedIndices);

  selectedBoundingBox = bbox.cols(selectedIndices);
  REQUIRE(selectedBoundingBox.n_cols == 2);
  REQUIRE(selectedBoundingBox.n_rows == 4);
  CheckMatrices(desiredBoundingBox, selectedBoundingBox);

  // Clean up.
  bbox.clear();
  desiredBoundingBox.clear();
  selectedBoundingBox.clear();

  // Use coordinate system to represent bounding boxes.
  // Bounding boxes represent {x0, y0, h, w}.
  bbox1 << 39 << 63 << 164 << 49;
  bbox2 << 31 << 69 << 170 << 56;
  bbox3 << 54 << 66 << 144 << 48;

  // Fill bounding box.
  bbox.insert_cols(0, bbox3);
  bbox.insert_cols(0, bbox2);
  bbox.insert_cols(0, bbox1);

  // Fill confidence scores of bounding boxes.
  confidenceScores << 1.0 << 0.6 << 0.4;

  // Selected bounding box using torchvision.ops.nms().
  desiredBoundingBox.insert_cols(0, bbox2);
  desiredBoundingBox.insert_cols(0, bbox1);

  NMS<false>::Evaluate(bbox, confidenceScores,
    selectedIndices, 0.7);

  selectedBoundingBox = bbox.cols(selectedIndices);
  REQUIRE(selectedBoundingBox.n_cols == 2);
  REQUIRE(selectedBoundingBox.n_rows == 4);
  CheckMatrices(desiredBoundingBox, selectedBoundingBox);
}

/**
 *
 */
TEST_CASE("BLEUScoreTest", "[MetricTest]")
{
  typedef typename std::vector<std::string> WordVector;
  std::vector<std::vector<WordVector>> referenceCorpus
      = {{{"this", "is", "my", "house"},
          {"this", "is", "my", "car"},
          {"this", "is", "my", "bike"}},

         {{"this", "is", "my", "table"},
          {"this", "is", "my", "chair"},
          {"this", "is", "my", "laptop"}},

         {{"this", "is", "my", "table"},
          {"this", "is", "your", "car"},
          {"this", "is", "my", "notebook"}}};

  std::vector<WordVector> translationCorpus
      = {{"this", "is", "my", "book"},
         {"this", "is", "your", "car"},
         {"this", "is", "my", "watch"}};

  BLEU<> bleu(4);

  //! We are not using smoothing function here.
  bleu.Evaluate(referenceCorpus, translationCorpus);
  REQUIRE(bleu.BLEUScore() == Approx(0.0).epsilon(1e-5));
  REQUIRE(bleu.BrevityPenalty() == 1.0);
  REQUIRE(bleu.Ratio() == 1.0);
  REQUIRE(bleu.TranslationLength() == 12);
  REQUIRE(bleu.ReferenceLength() == 12);

  std::vector<float> expectedPrecision = {0.666666f, 0.5555555f,
      0.3333333f, 0.0f};
  for (size_t i = 0; i < bleu.Precisions().size(); ++i)
  {
    REQUIRE(bleu.Precisions()[i] ==
        Approx((double)expectedPrecision[i]).epsilon(1e-4));
  }

  //! We will use smoothing function here by setting smooth to true.
  bleu.Evaluate(referenceCorpus, translationCorpus, true);
  REQUIRE(bleu.BLEUScore() == Approx(0.459307).epsilon(1e-5));
  REQUIRE(bleu.BrevityPenalty() == 1.0);
  REQUIRE(bleu.Ratio() == 1.0);
  REQUIRE(bleu.TranslationLength() == 12);
  REQUIRE(bleu.ReferenceLength() == 12);

  expectedPrecision = {0.692308f, 0.6f, 0.428571f, 0.25f};
  for (size_t i = 0; i < bleu.Precisions().size(); ++i)
  {
    REQUIRE(bleu.Precisions()[i] ==
        Approx(expectedPrecision[i]).epsilon(1e-4));
  }
}
