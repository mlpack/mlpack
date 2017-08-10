/**
 * @file hpt_test.cpp
 *
 * Tests for the hyper-parameter tuning module.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core/cv/metrics/mse.hpp>
#include <mlpack/core/cv/metrics/accuracy.hpp>
#include <mlpack/core/cv/simple_cv.hpp>
#include <mlpack/core/hpt/bind.hpp>
#include <mlpack/core/hpt/cv_function.hpp>
#include <mlpack/core/hpt/hpt.hpp>
#include <mlpack/core/optimizers/grid_search/grid_search.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack::cv;
using namespace mlpack::hpt;
using namespace mlpack::optimization;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(HPTTest);

/**
 * Test CVFunction runs cross-validation in according with specified bound
 * arguments and passed parameters.
 */
BOOST_AUTO_TEST_CASE(CVFunctionTest)
{
  arma::mat xs = arma::randn(5, 100);
  arma::vec beta = arma::randn(5, 1);
  arma::rowvec ys = beta.t() * xs + 0.1 * arma::randn(1, 100);

  SimpleCV<LARS, MSE> cv(0.2, xs, ys);

  bool transposeData = true;
  bool useCholesky = false;
  double lambda1 = 1.0;
  double lambda2 = 2.0;

  BoundArg<bool, 1> boundUseCholesky{useCholesky};
  BoundArg<double, 3> boundLambda1{lambda2};
  CVFunction<decltype(cv), LARS, 4, BoundArg<bool, 1>, BoundArg<double, 3>>
      cvFun(cv, boundUseCholesky, boundLambda1);

  double expected = cv.Evaluate(transposeData, useCholesky, lambda1, lambda2);
  double actual = cvFun.Evaluate(arma::vec{double(transposeData), lambda1});

  BOOST_REQUIRE_CLOSE(expected, actual, 1e-5);
}

void InitProneToOverfittingData(arma::mat& xs,
                                arma::rowvec& ys,
                                double& validationSize)
{
  // Making the function generate the same data for all callers.
  arma::arma_rng::set_seed(11);

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

 /**
 * Test grid-search optimization leads to the best parameters from the specified
 * ones.
 */
BOOST_AUTO_TEST_CASE(GridSearchTest)
{
  arma::mat xs;
  arma::rowvec ys;
  double validationSize;
  InitProneToOverfittingData(xs, ys, validationSize);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set =
      arma::join_cols(arma::vec{0}, arma::logspace<arma::vec>(-3, 2, 6));
  std::array<double, 4> lambda2Set{{0.0, 0.05, 0.5, 5.0}};

  double expectedLambda1, expectedLambda2, expectedObjective;
  FindLARSBestLambdas(xs, ys, validationSize, transposeData, useCholesky,
      lambda1Set, lambda2Set, expectedLambda1, expectedLambda2,
      expectedObjective);

  // We should get these values (it has been found empirically).
  BOOST_REQUIRE_CLOSE(expectedLambda1, 0.01, 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda2, 0.05, 1e-5);

  SimpleCV<LARS, MSE> cv(validationSize, xs, ys);

  GridSearch optimizer(lambda1Set, lambda2Set);
  CVFunction<decltype(cv), LARS, 4, BoundArg<bool, 0>, BoundArg<bool, 1>>
      cvFun(cv, {transposeData}, {useCholesky});
  arma::mat actualParameters;
  double actualObjective = optimizer.Optimize(cvFun, actualParameters);

  BOOST_REQUIRE_CLOSE(expectedObjective, actualObjective, 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda1, actualParameters(0, 0), 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda2, actualParameters(1, 0), 1e-5);
}

/**
 * Test HyperParameterTuner.
 */
BOOST_AUTO_TEST_CASE(HPTTest)
{
  arma::mat xs;
  arma::rowvec ys;
  double validationSize;
  InitProneToOverfittingData(xs, ys, validationSize);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set =
      arma::join_cols(arma::vec{0}, arma::logspace<arma::vec>(-3, 2, 6));
  arma::vec lambda2Set{0.0, 0.05, 0.5, 5.0};

  double expectedLambda1, expectedLambda2, expectedObjective;
  FindLARSBestLambdas(xs, ys, validationSize, transposeData, useCholesky,
      lambda1Set, lambda2Set, expectedLambda1, expectedLambda2,
      expectedObjective);

  // We should get these values (it has been found empirically).
  BOOST_REQUIRE_CLOSE(expectedLambda1, 0.01, 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda2, 0.05, 1e-5);

  double actualLambda1, actualLambda2;
  HyperParameterTuner<LARS, MSE, SimpleCV, GridSearch>
      hpt(validationSize, xs, ys);
  std::tie(actualLambda1, actualLambda2) = hpt.Optimize(Bind(transposeData),
      Bind(useCholesky), lambda1Set, lambda2Set);

  BOOST_REQUIRE_CLOSE(expectedObjective, hpt.BestObjective(), 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda1, actualLambda1, 1e-5);
  BOOST_REQUIRE_CLOSE(expectedLambda2, actualLambda2, 1e-5);

  /* Checking that the model provided by the hyper-parameter tuner shows the
   * same performance. */
  size_t validationFirstColumn = round(xs.n_cols * (1.0 - validationSize));
  arma::mat validationXs = xs.cols(validationFirstColumn, xs.n_cols - 1);
  arma::rowvec validationYs = ys.cols(validationFirstColumn, ys.n_cols - 1);
  double objective = MSE::Evaluate(hpt.BestModel(), validationXs, validationYs);
  BOOST_REQUIRE_CLOSE(expectedObjective, objective, 1e-5);
}

/**
 * Test HyperParamterTuner maximizes Accuracy rather than minimizes it.
 */
BOOST_AUTO_TEST_CASE(HPTMaximizationTest)
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
  arma::vec lambdas{0.0, 1e12};

  // Making sure that the assumption above is true.
  SimpleCV<LogisticRegression<>, Accuracy>
      cv(validationSize, doubledXs, doubledYs);
  BOOST_REQUIRE_GT(cv.Evaluate(0.0), cv.Evaluate(1e12));

  HyperParameterTuner<LogisticRegression<>, Accuracy, SimpleCV>
      hpt(validationSize, doubledXs, doubledYs);

  double actualLambda;
  std::tie(actualLambda) = hpt.Optimize(lambdas);

  BOOST_REQUIRE_CLOSE(hpt.BestObjective(), 1.0, 1e-5);
  BOOST_REQUIRE_CLOSE(actualLambda, 0.0, 1e-5);
}

BOOST_AUTO_TEST_SUITE_END();
