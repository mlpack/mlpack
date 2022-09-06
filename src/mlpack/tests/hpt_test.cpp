/**
 * @file tests/hpt_test.cpp
 *
 * Tests for the hyper-parameter tuning module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/lars.hpp>
#include <mlpack/methods/logistic_regression.hpp>

#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace ens;

/**
 * Test CVFunction runs cross-validation in according with specified fixed
 * arguments and passed parameters.
 */
TEST_CASE("CVFunctionTest", "[HPTTest]")
{
  arma::mat xs = arma::randn(5, 100);
  arma::vec beta = arma::randn(5, 1);
  arma::rowvec ys = beta.t() * xs + 0.1 * arma::randn(1, 100);

  SimpleCV<LARS, MSE> cv(0.2, xs, ys);

  bool transposeData = true;
  bool useCholesky = false;
  double lambda1 = 1.0;
  double lambda2 = 2.0;

  IncrementPolicy policy(true);
  DatasetMapper<IncrementPolicy, double> datasetInfo(policy, 2);

  FixedArg<bool, 1> fixedUseCholesky{useCholesky};
  FixedArg<double, 3> fixedLambda1{lambda2};
  CVFunction<decltype(cv), LARS, 4, FixedArg<bool, 1>, FixedArg<double, 3>>
      cvFun(cv, datasetInfo, 0.0, 0.0, fixedUseCholesky, fixedLambda1);

  double expected = cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
  arma::vec parameters(2);
  parameters(0) = transposeData;
  parameters(1) = lambda1;
  double actual = cvFun.Evaluate(parameters);

  REQUIRE(expected == Approx(actual).epsilon(1e-7));
}

/**
 * Test CVFunction runs cross-validation in according with specified fixed
 * arguments and passed parameters, where the passed parameters are categorical
 * parameters.
 */
TEST_CASE("CVFunctionCategoricalTest", "[HPTTest]")
{
  arma::mat xs = arma::randn(5, 100);
  arma::vec beta = arma::randn(5, 1);
  arma::rowvec ys = beta.t() * xs + 0.1 * arma::randn(1, 100);

  SimpleCV<LARS, MSE> cv(0.2, xs, ys);

  bool transposeData = true;
  bool useCholesky = false;
  double lambda1 = 1.0;
  double lambda2 = 2.0;

  IncrementPolicy policy(true);
  DatasetMapper<IncrementPolicy, double> datasetInfo(policy, 2);
  datasetInfo.MapString<double>(transposeData, 0);
  datasetInfo.MapString<double>(lambda1, 1);

  FixedArg<bool, 1> fixedUseCholesky{useCholesky};
  FixedArg<double, 3> fixedLambda1{lambda2};
  CVFunction<decltype(cv), LARS, 4, FixedArg<bool, 1>, FixedArg<double, 3>>
      cvFun(cv, datasetInfo, 0.0, 0.0, fixedUseCholesky, fixedLambda1);

  double expected = cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
  arma::vec parameters(2);
  parameters(0) = 0; // Should be unmapped to 'true'.
  parameters(1) = 0; // Should be unmapped to 1.0.
  double actual = cvFun.Evaluate(parameters);

  REQUIRE(expected == Approx(actual).epsilon(1e-7));
}

/**
 * This class provides the interface of CV classes, but really implements a
 * simple quadratic function of three variables.
 */
template<typename MLAlgorithm,
         typename Metric = void,
         typename MatType = void,
         typename PredictionsType = void,
         typename WeightsType = void>
class QuadraticFunction
{
 public:
  QuadraticFunction(double a,
                    double b,
                    double c,
                    double d,
                    double xMin = 0.0,
                    double yMin = 0.0,
                    double zMin = 0.0) :
      a(a), b(b), c(c), d(d), xMin(xMin), yMin(yMin), zMin(zMin) {}

  double Evaluate(double x, double y, double z)
  {
    return a * pow(x - xMin, 2)  + b * pow(y - yMin, 2) + c * pow(z - zMin, 2)
        + d;
  }

  // Declaring and defining it just in order to provide the same interface as
  // other CV classes.
  MLAlgorithm Model()
  {
    return MLAlgorithm();
  }

 private:
  double a, b, c, d, xMin, yMin, zMin;
};

/**
 * Test CVFunction approximates gradient in the expected way.
 */
TEST_CASE("CVFunctionGradientTest", "[HPTTest]")
{
  double a = 1.0;
  double b = -1.5;
  double c = 2.5;
  double d = 3.0;
  QuadraticFunction<LARS> lf(a, b, c, d);

  // All values are numeric.
  IncrementPolicy policy(true);
  DatasetMapper<IncrementPolicy, double> datasetInfo(policy, 3);

  double relativeDelta = 0.01;
  double minDelta = 0.001;
  CVFunction<decltype(lf), LARS, 3> cvFun(lf, datasetInfo, relativeDelta,
      minDelta);

  double x = 0.0;
  double y = -1.0;
  double z = 2.0;
  arma::mat gradient;
  cvFun.Gradient(arma::vec("0.0 -1.0 2.0"), gradient);

  double xDelta = minDelta;
  double yDelta = relativeDelta * abs(y);
  double zDelta = relativeDelta * abs(z);

  double aproximateXPartialDerivative = a * (2 * x + xDelta);
  double aproximateYPartialDerivative = b * (2 * y + yDelta);
  double aproximateZPartialDerivative = c * (2 * z + zDelta);

  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient(0) == Approx(aproximateXPartialDerivative).epsilon(1e-7));
  REQUIRE(gradient(1) == Approx(aproximateYPartialDerivative).epsilon(1e-7));
  REQUIRE(gradient(2) == Approx(aproximateZPartialDerivative).epsilon(1e-7));
}


void InitProneToOverfittingData(arma::mat& xs,
                                arma::rowvec& ys,
                                double& validationSize)
{
  // Total number of data points.
  size_t N = 10;
  // Total number of features (all except the first one are redundant).
  size_t M = 5;

  arma::rowvec data = arma::linspace<arma::rowvec>(0.0, 10.0, N);
  xs = data;
  for (size_t i = 2; i <= M; ++i)
    xs = arma::join_cols(xs, arma::pow(data, i));

  // Responses that approximately follow the function y = 2 * x. Adding noise to
  // avoid having a polynomial of degree 1 that exactly fits the points.
  ys = 2 * data + 0.05 * arma::randn(1, N);

  validationSize = 0.3;
}

template<typename T1, typename T2>
void FindLARSBestLambdas(arma::mat& xs,
                         arma::rowvec& ys,
                         double& validationSize,
                         bool transposeData,
                         bool useCholesky,
                         const T1& lambda1Set,
                         const T2& lambda2Set,
                         double& bestLambda1,
                         double& bestLambda2,
                         double& bestObjective)
{
  SimpleCV<LARS, MSE> cv(validationSize, xs, ys);

  bestObjective = std::numeric_limits<double>::max();

  for (double lambda1 : lambda1Set)
  {
    for (double lambda2 : lambda2Set)
    {
      double objective =
          cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
      if (objective < bestObjective)
      {
        bestObjective = objective;
        bestLambda1 = lambda1;
        bestLambda2 = lambda2;
      }
    }
  }
}

 /**
 * Test grid-search optimization leads to the best parameters from the specified
 * ones.
 */
TEST_CASE("GridSearchTest", "[HPTTest]")
{
  arma::mat xs;
  arma::rowvec ys;
  double validationSize;
  InitProneToOverfittingData(xs, ys, validationSize);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set("0 0.001 0.01 0.1 1.0 10.0 100.0");
  std::array<double, 4> lambda2Set{{0.0, 0.05, 0.5, 5.0}};

  double expectedLambda1, expectedLambda2, expectedObjective;
  FindLARSBestLambdas(xs, ys, validationSize, transposeData, useCholesky,
      lambda1Set, lambda2Set, expectedLambda1, expectedLambda2,
      expectedObjective);

  IncrementPolicy policy(true);
  DatasetMapper<IncrementPolicy, double> datasetInfo(policy, 2);
  for (double lambda1 : lambda1Set)
    datasetInfo.MapString<size_t>(lambda1, 0);
  for (double lambda2 : lambda2Set)
    datasetInfo.MapString<size_t>(lambda2, 1);

  SimpleCV<LARS, MSE> cv(validationSize, xs, ys);
  CVFunction<decltype(cv), LARS, 4, FixedArg<bool, 0>, FixedArg<bool, 1>>
      cvFun(cv, datasetInfo, 0.0, 0.0, {transposeData}, {useCholesky});

  ens::GridSearch optimizer;
  arma::mat actualParameters;

  std::vector<bool> categoricalDimensions(datasetInfo.Dimensionality());
  arma::Row<size_t> numCategories(datasetInfo.Dimensionality());
  for (size_t d = 0; d < datasetInfo.Dimensionality(); d++)
  {
    numCategories[d] = datasetInfo.NumMappings(d);
    categoricalDimensions[d] = datasetInfo.Type(d) ==
        mlpack::data::Datatype::categorical;
  }

  double actualObjective = optimizer.Optimize(cvFun, actualParameters,
      categoricalDimensions, numCategories);

  REQUIRE(expectedObjective == Approx(actualObjective).epsilon(1e-7));
  REQUIRE(expectedLambda1 ==
      Approx(datasetInfo.UnmapString(actualParameters(0, 0), 0)).epsilon(1e-7));
  REQUIRE(expectedLambda2 ==
      Approx(datasetInfo.UnmapString(actualParameters(1, 0), 1)).epsilon(1e-7));
}

/**
 * Test HyperParameterTuner.
 */
TEST_CASE("HPTTest", "[HPTTest]")
{
  arma::mat xs;
  arma::rowvec ys;
  double validationSize;
  InitProneToOverfittingData(xs, ys, validationSize);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set("0 0.001 0.01 0.1 1.0 10.0 100.0");
  arma::vec lambda2Set("0.0 0.05 0.5 5.0");

  double expectedLambda1, expectedLambda2, expectedObjective;
  FindLARSBestLambdas(xs, ys, validationSize, transposeData, useCholesky,
      lambda1Set, lambda2Set, expectedLambda1, expectedLambda2,
      expectedObjective);

  double actualLambda1, actualLambda2;
  HyperParameterTuner<LARS, MSE, SimpleCV, GridSearch>
      hpt(validationSize, xs, ys);
  std::tie(actualLambda1, actualLambda2) = hpt.Optimize(Fixed(transposeData),
      Fixed(useCholesky), lambda1Set, lambda2Set);

  REQUIRE(expectedObjective == Approx(hpt.BestObjective()).epsilon(1e-7));
  REQUIRE(expectedLambda1 == Approx(actualLambda1).epsilon(1e-7));
  REQUIRE(expectedLambda2 == Approx(actualLambda2).epsilon(1e-7));

  /* Checking that the model provided by the hyper-parameter tuner shows the
   * same performance. */
  size_t validationFirstColumn = round(xs.n_cols * (1.0 - validationSize));
  arma::mat validationXs = xs.cols(validationFirstColumn, xs.n_cols - 1);
  arma::rowvec validationYs = ys.cols(validationFirstColumn, ys.n_cols - 1);
  double objective = MSE::Evaluate(hpt.BestModel(), validationXs, validationYs);
  REQUIRE(expectedObjective == Approx(objective).epsilon(1e-7));
}

/**
 * Test HyperParamterTuner maximizes Accuracy rather than minimizes it.
 */
TEST_CASE("HPTMaximizationTest", "[HPTTest]")
{
  // Initializing a linearly separable dataset.
  arma::mat xs = arma::linspace<arma::rowvec>(0.0, 10.0, 50);
  arma::Row<size_t> ys = arma::join_rows(arma::zeros<arma::Row<size_t>>(25),
      arma::ones<arma::Row<size_t>>(25));

  // We will train and validate on the same dataset.
  double validationSize = 0.5;
  arma::mat doubledXs = arma::join_rows(xs, xs);
  arma::Row<size_t> doubledYs = arma::join_rows(ys, ys);

  // Defining lambdas to choose from. Zero should be preferred since big lambdas
  // are likely to restrict capabilities of logistic regression.
  arma::vec lambdas("0 1e12");

  // Making sure that the assumption above is true.
  SimpleCV<LogisticRegression<>, Accuracy>
      cv(validationSize, doubledXs, doubledYs);
  REQUIRE(cv.Evaluate(0.0) > cv.Evaluate(1e12));

  HyperParameterTuner<LogisticRegression<>, Accuracy, SimpleCV>
      hpt(validationSize, doubledXs, doubledYs);

  double actualLambda;
  std::tie(actualLambda) = hpt.Optimize(lambdas);

  REQUIRE(hpt.BestObjective() == Approx(1.0).epsilon(1e-7));
  REQUIRE(actualLambda == Approx(0.0).epsilon(1e-7));
}

/**
 * Test HyperParameterTuner works with GradientDescent.
 */
TEST_CASE("HPTGradientDescentTest", "[HPTTest]")
{
  // Constructor arguments for the fake CV function (QuadraticFunction).
  double a = 1.0;
  double b = -1.5;
  double c = 2.5;
  double d = 3.0;

  // Optimal values for three "hyper-parameters".
  double xMin = 1.5;
  double yMin = 0.0;
  double zMin = -2.0;

  // We pass LARS just because some ML algorithm should be passed. We pass MSE
  // to tell HyperParameterTuner that the objective function (QuadraticFunction)
  // should be minimized.
  HyperParameterTuner<LARS, MSE, QuadraticFunction,
      GradientDescent> hpt(a, b, c, d, xMin, yMin, zMin);

  // Setting GradientDescent to find more close solution to the optimal one.
  hpt.Optimizer().StepSize() = 0.1;
  hpt.Optimizer().Tolerance() = 1e-15;

  // Always using the same small increase of arguments in calculation of partial
  // derivatives.
  hpt.RelativeDelta() = 0.0;
  hpt.MinDelta() = 1e-10;

  // We will try to find optimal values only for two "hyper-parameters".
  double x0 = 3.0;
  double y = yMin;
  double z0 = -3.0;

  double xOptimized, zOptimized;
  std::tie(xOptimized, zOptimized) = hpt.Optimize(x0, Fixed(y), z0);
  REQUIRE(xOptimized == Approx(xMin).epsilon(1e-6));
  REQUIRE(zOptimized == Approx(zMin).epsilon(1e-6));
}
