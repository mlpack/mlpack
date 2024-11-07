/**
 * @file tests/lmnn_test.cpp
 * @author Marcus Edel
 * @author Ryan Curtin
 * @author Manish Kumar
 *
 * Unit tests for Large Margin Nearest Neighbors and related code (including
 * the constraints class).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/lmnn.hpp>
#include <mlpack/methods/neighbor_search.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;
using namespace ens;

//
// Tests for the Constraints.
//

/**
 * The target neighbors function should be correct.
 * point.
 */
TEMPLATE_TEST_CASE("LMNNTargetNeighborsTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  Constraints<arma::Mat<ElemType>, arma::Row<size_t>> constraint(dataset,
      labels, 1);

  // Calculate norm of datapoints.
  arma::Col<ElemType> norm(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    norm(i) = arma::norm(dataset.col(i));
  }

  //! Store target neighbors of data points.
  arma::umat targetNeighbors(1, dataset.n_cols);

  constraint.TargetNeighbors(targetNeighbors, dataset, labels, norm);

  REQUIRE(targetNeighbors(0, 0) == 1);
  REQUIRE(targetNeighbors(0, 1) == 0);
  REQUIRE(targetNeighbors(0, 2) == 1);
  REQUIRE(targetNeighbors(0, 3) == 4);
  REQUIRE(targetNeighbors(0, 4) == 3);
  REQUIRE(targetNeighbors(0, 5) == 4);
}

/**
 * The impostors function should be correct.
 */
TEMPLATE_TEST_CASE("LMNNImpostorsTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  Constraints<arma::Mat<ElemType>, arma::Row<size_t>> constraint(dataset,
      labels, 1);

  // Calculate norm of datapoints.
  arma::Col<ElemType> norm(dataset.n_cols);
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    norm(i) = arma::norm(dataset.col(i));
  }

  //! Store impostors of data points.
  arma::umat impostors(1, dataset.n_cols);

  constraint.Impostors(impostors, dataset, labels, norm);

  REQUIRE(impostors(0, 0) == 3);
  REQUIRE(impostors(0, 1) == 4);
  REQUIRE(impostors(0, 2) == 5);
  REQUIRE(impostors(0, 3) == 0);
  REQUIRE(impostors(0, 4) == 1);
  REQUIRE(impostors(0, 5) == 2);
}

//
// Tests for the LMNNFunction
//

/**
 * The LMNN function should return the identity matrix as its initial
 * point.
 */
TEMPLATE_TEST_CASE("LMNNInitialPointTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Cheap fake dataset.
  arma::Mat<ElemType> dataset = arma::randu<arma::Mat<ElemType>>(5, 5);
  arma::Row<size_t> labels = "0 1 1 0 0";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.5, 1);

  // Verify the initial point is the identity matrix.
  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  const double margin = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;
  arma::Mat<ElemType> initialPoint = lmnnfn.GetInitialPoint();
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      if (row == col)
        REQUIRE(initialPoint(row, col) == Approx(1.0).epsilon(eps));
      else
        REQUIRE(initialPoint(row, col) == Approx(0.0).margin(margin));
    }
  }
}

/***
 * Ensure non-separable objective function is right.
 */
TEMPLATE_TEST_CASE("LMNNInitialEvaluationTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  ElemType objective = lmnnfn.Evaluate(arma::eye<arma::Mat<ElemType>>(2, 2));

  // Result calculated by hand.
  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  REQUIRE(objective == Approx(9.456).epsilon(eps));
}

/**
 * Ensure non-separable gradient function is right.
 */
TEMPLATE_TEST_CASE("LMNNInitialGradientTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::Mat<ElemType> gradient;
  arma::Mat<ElemType> coordinates = arma::eye<arma::Mat<ElemType>>(2, 2);
  lmnnfn.Gradient(coordinates, gradient);

  // Result calculated by hand.
  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  const double margin = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;
  REQUIRE(gradient(0, 0) == Approx(-0.288).epsilon(eps));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(12.0).epsilon(eps));
}

/***
 * Ensure non-separable EvaluateWithGradient function is right.
 */
TEMPLATE_TEST_CASE("LMNNInitialEvaluateWithGradientTest", "[LMNNTest]", float,
    double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::Mat<ElemType> gradient;
  arma::Mat<ElemType> coordinates = arma::eye<arma::Mat<ElemType>>(2, 2);
  ElemType objective = lmnnfn.EvaluateWithGradient(coordinates, gradient);

  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  const double margin = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  // Result calculated by hand.
  REQUIRE(objective == Approx(9.456).epsilon(eps));
  // Check Gradient
  REQUIRE(gradient(0, 0) == Approx(-0.288).epsilon(eps));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(12.0).epsilon(eps));
}

/**
 * Ensure the separable objective function is right.
 */
TEMPLATE_TEST_CASE("LMNNSeparableObjectiveTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  // Result calculated by hand.
  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  arma::Mat<ElemType> coordinates = arma::eye<arma::Mat<ElemType>>(2, 2);
  REQUIRE(lmnnfn.Evaluate(coordinates, 0, 1) == Approx(1.576).epsilon(eps));
  REQUIRE(lmnnfn.Evaluate(coordinates, 1, 1) == Approx(1.576).epsilon(eps));
  REQUIRE(lmnnfn.Evaluate(coordinates, 2, 1) == Approx(1.576).epsilon(eps));
  REQUIRE(lmnnfn.Evaluate(coordinates, 3, 1) == Approx(1.576).epsilon(eps));
  REQUIRE(lmnnfn.Evaluate(coordinates, 4, 1) == Approx(1.576).epsilon(eps));
  REQUIRE(lmnnfn.Evaluate(coordinates, 5, 1) == Approx(1.576).epsilon(eps));
}

/**
 * Ensure the separable gradient is right.
 */
TEMPLATE_TEST_CASE("LMNNSeparableGradientTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::Mat<ElemType> coordinates = arma::eye<arma::Mat<ElemType>>(2, 2);
  arma::Mat<ElemType> gradient(2, 2);

  lmnnfn.Gradient(coordinates, 0, gradient, 1);

  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  const double margin = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  lmnnfn.Gradient(coordinates, 1, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  lmnnfn.Gradient(coordinates, 2, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  lmnnfn.Gradient(coordinates, 3, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  lmnnfn.Gradient(coordinates, 4, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  lmnnfn.Gradient(coordinates, 5, gradient, 1);

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));
}

/**
 * Ensure the separable EvaluateWithGradient function is right.
 */
TEMPLATE_TEST_CASE("LMNNSeparableEvaluateWithGradientTest", "[LMNNTest]", float,
    double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  arma::Mat<ElemType> coordinates = arma::eye<arma::Mat<ElemType>>(2, 2);
  arma::Mat<ElemType> gradient(2, 2);

  ElemType objective = lmnnfn.EvaluateWithGradient(coordinates, 0, gradient, 1);

  const double eps = std::is_same_v<ElemType, float> ? 1e-4 : 1e-7;
  const double margin = std::is_same_v<ElemType, float> ? 1e-4 : 1e-5;

  REQUIRE(objective == Approx(1.576).epsilon(eps));

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  objective = lmnnfn.EvaluateWithGradient(coordinates, 1, gradient, 1);

  REQUIRE(objective == Approx(1.576).epsilon(eps));

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  objective = lmnnfn.EvaluateWithGradient(coordinates, 2, gradient, 1);

  REQUIRE(objective == Approx(1.576).epsilon(eps));

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  objective = lmnnfn.EvaluateWithGradient(coordinates, 3, gradient, 1);

  REQUIRE(objective == Approx(1.576).epsilon(eps));

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  objective = lmnnfn.EvaluateWithGradient(coordinates, 4, gradient, 1);

  REQUIRE(objective == Approx(1.576).epsilon(eps));

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));

  objective = lmnnfn.EvaluateWithGradient(coordinates, 5, gradient, 1);

  REQUIRE(objective == Approx(1.576).epsilon(eps));

  REQUIRE(gradient(0, 0) == Approx(-0.048).epsilon(eps));
  REQUIRE(gradient(0, 1) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 0) == Approx(0.0).margin(margin));
  REQUIRE(gradient(1, 1) == Approx(2.0).epsilon(eps));
}

// Check that final objective value using SGD optimizer is optimal.
TEMPLATE_TEST_CASE("LMNNSGDSimpleDatasetTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNN<> lmnn(1);

  arma::Mat<ElemType> outputMatrix;
  lmnn.LearnDistance(dataset, labels, outputMatrix);

  // Ensure that the objective function is better now.
  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  ElemType initObj = lmnnfn.Evaluate(arma::eye<arma::Mat<ElemType>>(2, 2));
  ElemType finalObj = lmnnfn.Evaluate(outputMatrix);

  // finalObj must be less than initObj.
  REQUIRE(finalObj < initObj);
}

// Check that final objective value using L-BFGS optimizer is optimal.
TEMPLATE_TEST_CASE("LMNNLBFGSSimpleDatasetTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNN lmnn(1);

  arma::Mat<ElemType> outputMatrix;
  ens::L_BFGS lbfgs;
  lmnn.LearnDistance(dataset, labels, outputMatrix, lbfgs);

  // Ensure that the objective function is better now.
  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  ElemType initObj = lmnnfn.Evaluate(arma::eye<arma::Mat<ElemType>>(2, 2));
  ElemType finalObj = lmnnfn.Evaluate(outputMatrix);

  // finalObj must be less than initObj.
  REQUIRE(finalObj < initObj);
}

template<typename MatType, typename LabelsType>
double KnnAccuracy(const MatType& dataset,
                   const LabelsType& labels,
                   const size_t k)
{
  using ElemType = typename MatType::elem_type;

  LabelsType uniqueLabels = arma::unique(labels);

  arma::Mat<size_t> neighbors;
  arma::Mat<ElemType> distances;

  NeighborSearch<NearestNeighborSort, EuclideanDistance, MatType> knn;

  knn.Train(dataset);
  knn.Search(k, neighbors, distances);

  // Keep count.
  size_t count = 0.0;

  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    arma::Col<ElemType> m;
    m.zeros(uniqueLabels.n_cols);

    for (size_t j = 0; j < k; ++j)
      m(labels(neighbors(j, i))) += 1 / std::pow(distances(j, i) + 1, 2);

    size_t index = ConvTo<size_t>::From(arma::find(m == arma::max(m)));

    // Increase count if labels match.
    if (index == labels(i))
        count++;
  }

  // Return accuracy.
  return ((double) count / dataset.n_cols) * 100;
}

// Check that final accuracy is greater than initial accuracy on
// simple dataset.
TEMPLATE_TEST_CASE("LMNNAccuracyTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  // Taking k = 3 as the case of k = 1 can be easily observed.
  double initAccuracy = KnnAccuracy(dataset, labels, 3);

  LMNN<> lmnn(2);

  arma::Mat<ElemType> outputMatrix;
  lmnn.LearnDistance(dataset, labels, outputMatrix);

  arma::Mat<ElemType> transformedData = outputMatrix * dataset;
  double finalAccuracy = KnnAccuracy(transformedData, labels, 3);

  // finalObj must be less than initObj.
  REQUIRE(initAccuracy < finalAccuracy);

  // Since this is a very simple dataset final accuracy should be around 100%.
  REQUIRE(finalAccuracy == Approx(100.0).epsilon(1e-7));
}

// Check that accuracy while learning square distance matrix is the same as when
// we are learning low rank matrix.  I'm ok if this passes only once out of
// three tries.
TEMPLATE_TEST_CASE("LMNNLowRankAccuracyLBFGSTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::Mat<ElemType> dataPart1;
    dataPart1.randn(5, 50);

    arma::Row<size_t> labelsPart1(50);
    labelsPart1.fill(0);

    arma::Mat<ElemType> dataPart2;
    dataPart2.randn(5, 50);

    arma::Row<size_t> labelsPart2(50);
    labelsPart2.fill(1);

    // Generate ordering.
    arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0, 99, 100));

    // Generate datasets.
    arma::Mat<ElemType> dataset = join_rows(dataPart1, dataPart2);
    dataset = dataset.cols(ordering);

    // Generate labels.
    arma::Row<size_t> labels = join_rows(labelsPart1, labelsPart2);
    labels = labels.cols(ordering);

    LMNN<SquaredEuclideanDistance> lmnn(1);

    // Learn a square matrix.
    arma::Mat<ElemType> outputMatrix;
    L_BFGS lbfgs;
    lmnn.LearnDistance(dataset, labels, outputMatrix, lbfgs);

    arma::Mat<ElemType> transformedData = outputMatrix * dataset;
    double acc1 = KnnAccuracy(transformedData, labels, 1);

    // Learn a low rank matrix.
    outputMatrix = arma::randu<arma::Mat<ElemType>>(4, 5);
    lmnn.LearnDistance(dataset, labels, outputMatrix, lbfgs);

    transformedData = outputMatrix * dataset;
    double acc2 = KnnAccuracy(transformedData, labels, 1);

    // We keep the tolerance very high.  We need to ensure the accuracy drop
    // isn't any more than 10%.
    success = ((acc1 - acc2) <= 10.0);
    if (success)
      break;
  }

  REQUIRE(success == true);
}

// Check that accuracy while learning square distance matrix is the same as when
// we are learning low rank matrix.  I'm ok if this passes only once out of
// three tries.
TEMPLATE_TEST_CASE("LMNNLowRankAccuracyTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  bool success = false;
  for (size_t trial = 0; trial < 3; ++trial)
  {
    arma::Mat<ElemType> dataPart1;
    dataPart1.randn(5, 50);

    arma::Row<size_t> labelsPart1(50);
    labelsPart1.fill(0);

    arma::Mat<ElemType> dataPart2;
    dataPart2.randn(5, 50);

    arma::Row<size_t> labelsPart2(50);
    labelsPart2.fill(1);

    // Generate ordering.
    arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0, 99, 100));

    // Generate datasets.
    arma::Mat<ElemType> dataset = join_rows(dataPart1, dataPart2);
    dataset = dataset.cols(ordering);

    // Generate labels.
    arma::Row<size_t> labels = join_rows(labelsPart1, labelsPart2);
    labels = labels.cols(ordering);

    LMNN<> lmnn(1);

    // Learn a square matrix.
    arma::Mat<ElemType> outputMatrix;
    lmnn.LearnDistance(dataset, labels, outputMatrix);

    arma::Mat<ElemType> transformedData = outputMatrix * dataset;
    double acc1 = KnnAccuracy(transformedData, labels, 1);

    // Learn a low rank matrix.
    outputMatrix = arma::randu<arma::Mat<ElemType>>(4, 5);
    lmnn.LearnDistance(dataset, labels, outputMatrix);

    transformedData = outputMatrix * dataset;
    double acc2 = KnnAccuracy(transformedData, labels, 1);

    // We keep the tolerance very high.  We need to ensure the accuracy drop
    // isn't any more than 10%.
    success = ((acc1 - acc2) <= 10.0);
    if (success)
      break;
  }

  REQUIRE(success == true);
}

// Check that accuracy while learning square distance matrix is the same as when
// we are learning low rank matrix.  I'm ok if this passes only once out of
// five tries, since BBSGD seems to have a harder time converging.
/*
TEST_CASE("LMNNLowRankAccuracyBBSGDTest", "[LMNNTest]")
{
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat dataPart1;
    dataPart1.randn(5, 50);

    arma::Row<size_t> labelsPart1(50);
    labelsPart1.fill(0);

    arma::mat dataPart2;
    dataPart2.randn(5, 50);

    arma::Row<size_t> labelsPart2(50);
    labelsPart2.fill(1);

    // Generate ordering.
    arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0, 99, 100));

    // Generate datasets.
    arma::mat dataset = join_rows(dataPart1, dataPart2);
    dataset = dataset.cols(ordering);

    // Generate labels.
    arma::Row<size_t> labels = join_rows(labelsPart1, labelsPart2);
    labels = labels.cols(ordering);

    LMNN<SquaredEuclideanDistance, BigBatchSGD<>> lmnn(dataset, labels, 1);

    // Learn a square matrix.
    arma::mat outputMatrix;
    lmnn.LearnDistance(outputMatrix);

    double acc1 = KnnAccuracy(outputMatrix * dataset, labels, 1);

    // Learn a low rank matrix.
    outputMatrix = arma::randu(4, 5);
    lmnn.LearnDistance(outputMatrix);

    double acc2 = KnnAccuracy(outputMatrix * dataset, labels, 1);
    if (acc2 < 5)
      std::cout << "super fail\n" << outputMatrix << std::endl;

    // We keep the tolerance very high.  We need to ensure the accuracy drop
    // isn't any more than 10%.
    success = ((acc1 - acc2) <= 10.0);
    if (success)
      break;
  }

  REQUIRE(success == true);
}
*/

// Comprehensive gradient tests by Marcus Edel & Ryan Curtin.

// Simple numerical gradient checker.
template<typename FunctionType, typename MatType>
double CheckGradient(FunctionType& function,
                     MatType& coordinates,
                     const typename MatType::elem_type eps = 1e-7)
{
  using ElemType = typename MatType::elem_type;

  // Get gradients for the current parameters.
  MatType orgGradient, gradient, estGradient;
  function.Gradient(coordinates, orgGradient);

  estGradient = arma::zeros<MatType>(orgGradient.n_rows, orgGradient.n_cols);

  // Compute numeric approximations to gradient.
  for (size_t i = 0; i < orgGradient.n_elem; ++i)
  {
    ElemType tmp = coordinates(i);

    // Perturb parameter with a positive constant and get costs.
    coordinates(i) += eps;
    ElemType costPlus = function.Evaluate(coordinates);

    // Perturb parameter with a negative constant and get costs.
    coordinates(i) -= (2 * eps);
    ElemType costMinus = function.Evaluate(coordinates);

    // Restore the parameter value.
    coordinates(i) = tmp;

    // Compute numerical gradients using the costs calculated above.
    estGradient(i) = (costPlus - costMinus) / (2 * eps);
  }

  // Estimate error of gradient.
  return arma::norm(orgGradient - estGradient) /
      arma::norm(orgGradient + estGradient);
}

TEMPLATE_TEST_CASE("LMNNFunctionGradientTest", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::Mat<ElemType> coordinates(2, 2, arma::fill::randn);
    CheckGradient(lmnnfn, coordinates);
  }
}

TEMPLATE_TEST_CASE("LMNNFunctionGradientTest2", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  // Useful but simple dataset with six points and two classes.
  arma::Mat<ElemType> dataset = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                                " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels    = " 0    0    0    1    1    1   ";

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::Mat<ElemType> coordinates(2, 2, arma::fill::randu);
    CheckGradient(lmnnfn, coordinates);
  }
}

TEMPLATE_TEST_CASE("LMNNFunctionGradientTest3", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  arma::Mat<ElemType> dataset;
  arma::Row<size_t> labels;
  if (!data::Load("iris.csv", dataset))
    FAIL("Cannot load dataset iris.csv");
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load dataset iris_labels.txt");

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::Mat<ElemType> coordinates(dataset.n_rows, dataset.n_rows,
        arma::fill::randn);
    CheckGradient(lmnnfn, coordinates);
  }
}

TEMPLATE_TEST_CASE("LMNNFunctionGradientTest4", "[LMNNTest]", float, double)
{
  using ElemType = TestType;

  arma::Mat<ElemType> dataset;
  arma::Row<size_t> labels;
  if (!data::Load("iris.csv", dataset))
    FAIL("Cannot load dataset iris.csv");
  if (!data::Load("iris_labels.txt", labels))
    FAIL("Cannot load dataset iris_labels.txt");

  LMNNFunction<arma::Mat<ElemType>> lmnnfn(dataset, labels, 1, 0.6, 1);

  // 10 trials with random positions.
  for (size_t i = 0; i < 10; ++i)
  {
    arma::Mat<ElemType> coordinates(dataset.n_rows, dataset.n_rows,
        arma::fill::randu);
    CheckGradient(lmnnfn, coordinates);
  }
}
