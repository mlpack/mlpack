/**
 * @file tests/logistic_regression_test.cpp
 * @author Ryan Curtin
 * @author Arun Reddy
 *
 * Test for LogisticFunction and LogisticRegression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * Test the LogisticRegressionFunction on a simple set of points.
 */
TEST_CASE("LogisticRegressionFunctionEvaluate", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a LogisticRegressionFunction.
  LogisticRegressionFunction<> lrf(data, responses,
      0.0 /* no regularization */);

  // These were hand-calculated using Octave.
  REQUIRE(lrf.Evaluate(arma::rowvec("1 1 1")) ==
     Approx(7.0562141665).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("0 0 0")) ==
      Approx(2.0794415417).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("-1 -1 -1")) ==
      Approx(8.0562141665).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -40 -40")) ==
      Approx(0.0).margin(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -80 0")) ==
      Approx(0.0).margin(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -100 20")) ==
      Approx(0.0).margin(1e-7));
}

/**
 * A more complicated test for the LogisticRegressionFunction.
 */
TEST_CASE("LogisticRegressionFunctionRandomEvaluate",
          "[LogisticRegressionTest]")
{
  const size_t points = 1000;
  const size_t dimension = 10;
  const size_t trials = 50;

  // Create a random dataset.
  arma::mat data;
  data.randu(dimension, points);
  // Create random responses.
  arma::Row<size_t> responses(points);
  for (size_t i = 0; i < points; ++i)
    responses[i] = RandInt(0, 2);

  LogisticRegressionFunction<> lrf(data, responses,
      0.0 /* no regularization */);

  // Run a bunch of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Generate a random set of parameters.
    arma::rowvec parameters;
    parameters.randu(dimension + 1);

    // Hand-calculate the loss function.
    double loglikelihood = 0.0;
    for (size_t j = 0; j < points; ++j)
    {
      const double sigmoid = (1.0 / (1.0 + exp(-parameters[0] -
          arma::dot(data.col(j), parameters.subvec(1, dimension)))));
      if (responses[j] == 1.0)
        loglikelihood += log(std::pow(sigmoid, responses[j]));
      else
        loglikelihood += log(std::pow(1.0 - sigmoid, 1.0 - responses[j]));
    }

    REQUIRE(lrf.Evaluate(parameters) == Approx(-loglikelihood).epsilon(1e-7));
  }
}

/**
 * Test regularization for the LogisticRegressionFunction Evaluate() function.
 */
TEST_CASE("LogisticRegressionFunctionRegularizationEvaluate",
          "[LogisticRegressionTest]")
{
  const size_t points = 5000;
  const size_t dimension = 25;
  const size_t trials = 10;

  // Create a random dataset.
  arma::mat data;
  data.randu(dimension, points);
  // Create random responses.
  arma::Row<size_t> responses(points);
  for (size_t i = 0; i < points; ++i)
    responses[i] = RandInt(0, 2);

  LogisticRegressionFunction<> lrfNoReg(data, responses, 0.0);
  LogisticRegressionFunction<> lrfSmallReg(data, responses, 0.5);
  LogisticRegressionFunction<> lrfBigReg(data, responses, 20.0);

  for (size_t i = 0; i < trials; ++i)
  {
    arma::rowvec parameters(dimension + 1);
    parameters.randu();

    // Regularization term: 0.5 * lambda * || parameters ||_2^2 (but note that
    // the first parameters term is ignored).
    const double smallRegTerm = 0.25 * std::pow(arma::norm(parameters, 2), 2.0)
        - 0.25 * std::pow(parameters[0], 2.0);
    const double bigRegTerm = 10.0 * std::pow(arma::norm(parameters, 2), 2.0)
        - 10.0 * std::pow(parameters[0], 2.0);

    REQUIRE(lrfNoReg.Evaluate(parameters) + smallRegTerm ==
        Approx(lrfSmallReg.Evaluate(parameters)).epsilon(1e-7));
    REQUIRE(lrfNoReg.Evaluate(parameters) + bigRegTerm ==
        Approx(lrfBigReg.Evaluate(parameters)).epsilon(1e-7));
  }
}

/**
 * Test gradient of the LogisticRegressionFunction.
 */
TEST_CASE("LogisticRegressionFunctionGradient", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a LogisticRegressionFunction.
  LogisticRegressionFunction<> lrf(data, responses,
      0.0 /* no regularization */);
  arma::rowvec gradient;

  // If the model is at the optimum, then the gradient should be zero.
  lrf.Gradient(arma::rowvec("200 -40 -40"), gradient);

  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  // Perturb two elements in the wrong way, so they need to become smaller.
  lrf.Gradient(arma::rowvec("200 -20 -20"), gradient);

  // The actual values are less important; the gradient just needs to be pointed
  // the right way.
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[1] >= 0.0);
  REQUIRE(gradient[2] >= 0.0);

  // Perturb two elements in the wrong way, so they need to become larger.
  lrf.Gradient(arma::rowvec("200 -60 -60"), gradient);

  // The actual values are less important; the gradient just needs to be pointed
  // the right way.
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[1] <= 0.0);
  REQUIRE(gradient[2] <= 0.0);

  // Perturb the intercept element.
  lrf.Gradient(arma::rowvec("250 -40 -40"), gradient);

  // The actual values are less important; the gradient just needs to be pointed
  // the right way.
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] >= 0.0);
}

/**
 * Test individual Evaluate() functions for SGD.
 */
TEST_CASE("LogisticRegressionSeparableEvaluate", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3;");
  arma::Row<size_t> responses("1 1 0");

  // Create a LogisticRegressionFunction.
  LogisticRegressionFunction<> lrf(data, responses,
      0.0 /* no regularization */);

  // These were hand-calculated using Octave.
  REQUIRE(lrf.Evaluate(arma::rowvec("1 1 1"), 0, 1) ==
      Approx(4.85873516e-2).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("1 1 1"), 1, 1) ==
      Approx(6.71534849e-3).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("1 1 1"), 2, 1) ==
      Approx(7.00091146645).epsilon(1e-7));

  REQUIRE(lrf.Evaluate(arma::rowvec("0 0 0"), 0, 1) ==
      Approx(0.6931471805).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("0 0 0"), 1, 1) ==
      Approx(0.6931471805).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("0 0 0"), 2, 1) ==
      Approx(0.6931471805).epsilon(1e-7));

  REQUIRE(lrf.Evaluate(arma::rowvec("-1 -1 -1"), 0, 1) ==
      Approx(3.0485873516).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("-1 -1 -1"), 1, 1) ==
      Approx(5.0067153485).epsilon(1e-7));
  REQUIRE(lrf.Evaluate(arma::rowvec("-1 -1 -1"), 2, 1) ==
      Approx(9.1146645377e-4).epsilon(1e-7));

  REQUIRE(lrf.Evaluate(arma::rowvec("200 -40 -40"), 0, 1) ==
      Approx(0.0).margin(1e-5));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -40 -40"), 1, 1) ==
      Approx(0.0).margin(1e-5));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -40 -40"), 2, 1) ==
      Approx(0.0).margin(1e-5));

  REQUIRE(lrf.Evaluate(arma::rowvec("200 -80 0"), 0, 1) ==
      Approx(0.0).margin(1e-5));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -80 0"), 1, 1) ==
      Approx(0.0).margin(1e-5));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -80 0"), 2, 1) ==
      Approx(0.0).margin(1e-5));

  REQUIRE(lrf.Evaluate(arma::rowvec("200 -100 20"), 0, 1) ==
      Approx(0.0).margin(1e-5));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -100 20"), 1, 1) ==
      Approx(0.0).margin(1e-5));
  REQUIRE(lrf.Evaluate(arma::rowvec("200 -100 20"), 2, 1) ==
      Approx(0.0).margin(1e-5));
}

/**
 * Test regularization for the separable LogisticRegressionFunction Evaluate()
 * function.
 */
TEST_CASE("LogisticRegressionFunctionRegularizationSeparableEvaluate",
          "[LogisticRegressionTest]")
{
  const size_t points = 5000;
  const size_t dimension = 25;
  const size_t trials = 10;

  // Create a random dataset.
  arma::mat data;
  data.randu(dimension, points);
  // Create random responses.
  arma::Row<size_t> responses(points);
  for (size_t i = 0; i < points; ++i)
    responses[i] = RandInt(0, 2);

  LogisticRegressionFunction<> lrfNoReg(data, responses, 0.0);
  LogisticRegressionFunction<> lrfSmallReg(data, responses, 0.5);
  LogisticRegressionFunction<> lrfBigReg(data, responses, 20.0);

  // Check that the number of functions is correct.
  REQUIRE(lrfNoReg.NumFunctions() == points);
  REQUIRE(lrfSmallReg.NumFunctions() == points);
  REQUIRE(lrfBigReg.NumFunctions() == points);

  for (size_t i = 0; i < trials; ++i)
  {
    arma::rowvec parameters(dimension + 1);
    parameters.randu();

    // Regularization term: 0.5 * lambda * || parameters ||_2^2 (but note that
    // the first parameters term is ignored).
    const double smallRegTerm = (0.25 * std::pow(arma::norm(parameters, 2), 2.0)
        - 0.25 * std::pow(parameters[0], 2.0)) / points;
    const double bigRegTerm = (10.0 * std::pow(arma::norm(parameters, 2), 2.0)
        - 10.0 * std::pow(parameters[0], 2.0)) / points;

    for (size_t j = 0; j < points; ++j)
    {
      REQUIRE(lrfNoReg.Evaluate(parameters, j, 1) + smallRegTerm ==
          Approx(lrfSmallReg.Evaluate(parameters, j, 1)).epsilon(1e-7));
      REQUIRE(lrfNoReg.Evaluate(parameters, j, 1) + bigRegTerm ==
          Approx(lrfBigReg.Evaluate(parameters, j, 1)).epsilon(1e-7));
    }
  }
}

/**
 * Test separable gradient of the LogisticRegressionFunction.
 */
TEST_CASE("LogisticRegressionFunctionSeparableGradient",
          "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a LogisticRegressionFunction.
  LogisticRegressionFunction<> lrf(data, responses,
      0.0 /* no regularization */);
  arma::rowvec gradient;

  // If the model is at the optimum, then the gradient should be zero.
  lrf.Gradient(arma::rowvec("200 -40 -40"), 0, gradient, 1);

  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  lrf.Gradient(arma::rowvec("200 -40 -40"), 1, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  lrf.Gradient(arma::rowvec("200 -40 -40"), 2, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  // Perturb two elements in the wrong way, so they need to become smaller.  For
  // the first two data points, classification is still correct so the gradient
  // should be zero.
  lrf.Gradient(arma::rowvec("200 -30 -30"), 0, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  lrf.Gradient(arma::rowvec("200 -30 -30"), 1, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  lrf.Gradient(arma::rowvec("200 -30 -30"), 2, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[1] >= 0.0);
  REQUIRE(gradient[2] >= 0.0);

  // Perturb two elements in the other wrong way, so they need to become larger.
  // For the first and last data point, classification is still correct so the
  // gradient should be zero.
  lrf.Gradient(arma::rowvec("200 -60 -60"), 0, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));

  lrf.Gradient(arma::rowvec("200 -30 -30"), 1, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[1] <= 0.0);
  REQUIRE(gradient[2] <= 0.0);

  lrf.Gradient(arma::rowvec("200 -60 -60"), 2, gradient, 1);
  REQUIRE(gradient.n_elem == 3);
  REQUIRE(gradient[0] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[1] == Approx(0.0).margin(1e-15));
  REQUIRE(gradient[2] == Approx(0.0).margin(1e-15));
}

/**
 * Test Gradient() function when regularization is used.
 */
TEST_CASE("LogisticRegressionFunctionRegularizationGradient",
          "[LogisticRegressionTest]")
{
  const size_t points = 5000;
  const size_t dimension = 25;
  const size_t trials = 10;

  // Create a random dataset.
  arma::mat data;
  data.randu(dimension, points);
  // Create random responses.
  arma::Row<size_t> responses(points);
  for (size_t i = 0; i < points; ++i)
    responses[i] = RandInt(0, 2);

  LogisticRegressionFunction<> lrfNoReg(data, responses, 0.0);
  LogisticRegressionFunction<> lrfSmallReg(data, responses, 0.5);
  LogisticRegressionFunction<> lrfBigReg(data, responses, 20.0);

  for (size_t i = 0; i < trials; ++i)
  {
    arma::rowvec parameters(dimension + 1);
    parameters.randu();

    // Regularization term: 0.5 * lambda * || parameters ||_2^2 (but note that
    // the first parameters term is ignored).  Now we take the gradient of this
    // to obtain
    //   g[i] = lambda * parameters[i]
    // although g(0) == 0 because we are not regularizing the intercept term of
    // the model.
    arma::rowvec gradient;
    arma::rowvec smallRegGradient;
    arma::rowvec bigRegGradient;

    lrfNoReg.Gradient(parameters, gradient);
    lrfSmallReg.Gradient(parameters, smallRegGradient);
    lrfBigReg.Gradient(parameters, bigRegGradient);

    // Check sizes of gradients.
    REQUIRE(gradient.n_elem == parameters.n_elem);
    REQUIRE(smallRegGradient.n_elem == parameters.n_elem);
    REQUIRE(bigRegGradient.n_elem == parameters.n_elem);

    // Make sure first term has zero regularization.
    REQUIRE(gradient[0] == Approx(smallRegGradient[0]).epsilon(1e-7));
    REQUIRE(gradient[0] == Approx(bigRegGradient[0]).epsilon(1e-7));

    // Check other terms.
    for (size_t j = 1; j < parameters.n_elem; ++j)
    {
      const double smallRegTerm = 0.5 * parameters[j];
      const double bigRegTerm = 20.0 * parameters[j];

      REQUIRE(gradient[j] + smallRegTerm == Approx(smallRegGradient[j]).
          epsilon(1e-7));
      REQUIRE(gradient[j] + bigRegTerm ==
          Approx(bigRegGradient[j]).epsilon(1e-7));
    }
  }
}

/**
 * Test separable Gradient() function when regularization is used.
 */
TEST_CASE("LogisticRegressionFunctionRegularizationSeparableGradient",
          "[LogisticRegressionTest]")
{
  const size_t points = 2000;
  const size_t dimension = 25;
  const size_t trials = 3;

  // Create a random dataset.
  arma::mat data;
  data.randu(dimension, points);
  // Create random responses.
  arma::Row<size_t> responses(points);
  for (size_t i = 0; i < points; ++i)
    responses[i] = RandInt(0, 2);

  LogisticRegressionFunction<> lrfNoReg(data, responses, 0.0);
  LogisticRegressionFunction<> lrfSmallReg(data, responses, 0.5);
  LogisticRegressionFunction<> lrfBigReg(data, responses, 20.0);

  for (size_t i = 0; i < trials; ++i)
  {
    arma::rowvec parameters(dimension + 1);
    parameters.randu();

    // Regularization term: 0.5 * lambda * || parameters ||_2^2 (but note that
    // the first parameters term is ignored).  Now we take the gradient of this
    // to obtain
    //   g[i] = lambda * parameters[i]
    // although g(0) == 0 because we are not regularizing the intercept term of
    // the model.
    arma::rowvec gradient;
    arma::rowvec smallRegGradient;
    arma::rowvec bigRegGradient;

    // Test separable gradient for each point.  Regularization will be the same.
    for (size_t k = 0; k < points; ++k)
    {
      lrfNoReg.Gradient(parameters, k, gradient, 1);
      lrfSmallReg.Gradient(parameters, k, smallRegGradient, 1);
      lrfBigReg.Gradient(parameters, k, bigRegGradient, 1);

      // Check sizes of gradients.
      REQUIRE(gradient.n_elem == parameters.n_elem);
      REQUIRE(smallRegGradient.n_elem == parameters.n_elem);
      REQUIRE(bigRegGradient.n_elem == parameters.n_elem);

      // Make sure first term has zero regularization.
      REQUIRE(gradient[0] == Approx(smallRegGradient[0]).epsilon(1e-7));
      REQUIRE(gradient[0] == Approx(bigRegGradient[0]).epsilon(1e-7));

      // Check other terms.
      for (size_t j = 1; j < parameters.n_elem; ++j)
      {
        const double smallRegTerm = 0.5 * parameters[j] / points;
        const double bigRegTerm = 20.0 * parameters[j] / points;

        REQUIRE(gradient[j] + smallRegTerm == Approx(smallRegGradient[j]).
            epsilon(1e-7));
        REQUIRE(gradient[j] + bigRegTerm ==
            Approx(bigRegGradient[j]).epsilon(1e-7));
      }
    }
  }
}

// Test training of logistic regression on a simple dataset.
TEST_CASE("LogisticRegressionLBFGSSimpleTest", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a logistic regression object using L-BFGS (that is the default).
  LogisticRegression<> lr(data, responses);

  // Test sigmoid function.
  arma::rowvec sigmoids = 1 / (1 + arma::exp(-lr.Parameters()[0]
      - lr.Parameters().tail_cols(lr.Parameters().n_elem - 1) * data));

  // Large 0.1% error tolerance is because the optimizer may terminate before
  // the predictions converge to 1.
  REQUIRE(sigmoids[0] == Approx(1.0).epsilon(1e-3));
  REQUIRE(sigmoids[1] == Approx(1.0).epsilon(0.05));
  REQUIRE(sigmoids[2] == Approx(0.0).margin(0.1));
}

// Test training of logistic regression on a simple dataset using SGD.
TEST_CASE("LogisticRegressionSGDSimpleTest", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a logistic regression object using a custom SGD object with a much
  // smaller tolerance.
  ens::StandardSGD sgd(0.005, 1, 500000, 1e-10);
  LogisticRegression<> lr(data, responses, sgd, 0.001);

  // Test sigmoid function.
  arma::rowvec sigmoids = 1 / (1 + arma::exp(-lr.Parameters()[0]
      - lr.Parameters().tail_cols(lr.Parameters().n_elem - 1) * data));

  // Large 0.1% error tolerance is because the optimizer may terminate before
  // the predictions converge to 1.  SGD tolerance is larger because its default
  // convergence tolerance is larger.
  REQUIRE(sigmoids[0] == Approx(1.0).epsilon(0.03));
  REQUIRE(sigmoids[1] == Approx(1.0).epsilon(0.12));
  REQUIRE(sigmoids[2] == Approx(0.0).margin(0.1));
}

// Test training of logistic regression on a simple dataset with regularization.
TEST_CASE("LogisticRegressionLBFGSRegularizationSimpleTest",
          "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a logistic regression object using L-BFGS (that is the default).
  LogisticRegression<> lr(data, responses, 0.001);

  // Test sigmoid function.
  arma::rowvec sigmoids = 1 / (1 + arma::exp(-lr.Parameters()[0]
      - lr.Parameters().tail_cols(lr.Parameters().n_elem - 1) * data));

  // Large error tolerance is because the optimizer may terminate before
  // the predictions converge to 1.
  REQUIRE(sigmoids[0] == Approx(1.0).epsilon(0.05));
  REQUIRE(sigmoids[1] == Approx(1.0).epsilon(0.10));
  REQUIRE(sigmoids[2] == Approx(0.0).margin(0.1));
}

// Test training of logistic regression on a simple dataset using SGD with
// regularization.
TEST_CASE("LogisticRegressionSGDRegularizationSimpleTest",
          "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create a logistic regression object using custom SGD with a much smaller
  // tolerance.
  ens::StandardSGD sgd(0.005, 32, 500000, 1e-10);
  LogisticRegression<> lr(data, responses, sgd, 0.001);

  // Test sigmoid function.
  arma::rowvec sigmoids = 1 / (1 + arma::exp(-lr.Parameters()[0]
      - lr.Parameters().tail_cols(lr.Parameters().n_elem - 1) * data));

  // Large error tolerance is because the optimizer may terminate before
  // the predictions converge to 1.  SGD tolerance is wider because its default
  // convergence tolerance is larger.
  REQUIRE(sigmoids[0] == Approx(1.0).epsilon(0.07));
  REQUIRE(sigmoids[1] == Approx(1.0).epsilon(0.14));
  REQUIRE(sigmoids[2] == Approx(0.0).margin(0.1));
}

// Test training of logistic regression on two Gaussians and ensure it's
// properly separable.
TEST_CASE("LogisticRegressionLBFGSGaussianTest", "[LogisticRegressionTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  lr.Train<ens::L_BFGS>(data, responses);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Ensure that the error is close to zero.
  const double testAcc = lr.ComputeAccuracy(data, responses);

  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

// Test training of logistic regression on two Gaussians and ensure it's
// properly separable using SGD.
TEST_CASE("LogisticRegressionSGDGaussianTest", "[LogisticRegressionTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  lr.Train<ens::StandardSGD>(data, responses);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);

  REQUIRE(acc == Approx(100.0).epsilon(0.003)); // 0.3% error tolerance.

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Ensure that the error is close to zero.
  const double testAcc = lr.ComputeAccuracy(data, responses);

  REQUIRE(testAcc == Approx(100.0).epsilon(0.006)); // 0.6% error tolerance.
}

/**
 * Test constructor that takes an already-instantiated optimizer.
 */
TEST_CASE("LogisticRegressionInstantiatedOptimizer", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Create an optimizer and function.
  ens::L_BFGS lbfgsOpt;
  lbfgsOpt.MinGradientNorm() = 1e-50;
  LogisticRegression<> lr(data, responses, lbfgsOpt, 0.0005);

  // Test sigmoid function.
  arma::rowvec sigmoids = 1 / (1 + arma::exp(-lr.Parameters()[0]
      - lr.Parameters().tail_cols(lr.Parameters().n_elem - 1) * data));

  // Error tolerance is small because we tightened the optimizer tolerance.
  REQUIRE(sigmoids[0] == Approx(1.0).epsilon(1e-3));
  REQUIRE(sigmoids[1] == Approx(1.0).epsilon(0.006));
  REQUIRE(sigmoids[2] == Approx(0.0).margin(0.1));

  // Now do the same with SGD.
  ens::StandardSGD sgdOpt;
  sgdOpt.StepSize() = 0.15;
  sgdOpt.Tolerance() = 1e-75;
  LogisticRegression<> lr2(data, responses, sgdOpt, 0.0005);

  // Test sigmoid function.
  sigmoids = 1 / (1 + arma::exp(-lr2.Parameters()[0]
      - lr2.Parameters().tail_cols(lr2.Parameters().n_elem - 1) * data));

  // Error tolerance is small because we tightened the optimizer tolerance.
  REQUIRE(sigmoids[0] == Approx(1.0).epsilon(1e-3));
  REQUIRE(sigmoids[1] == Approx(1.0).epsilon(0.006));
  REQUIRE(sigmoids[2] == Approx(0.0).margin(0.1));
}

/**
 * Test the Train() function and make sure it works the same as if we'd called
 * the constructor by hand, with the L-BFGS optimizer.
 */
TEST_CASE("LogisticRegressionLBFGSTrainTest", "[LogisticRegressionTest]")
{
  // Make a random dataset with random labels.
  arma::mat dataset(5, 800);
  dataset.randu();
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LogisticRegression<> lr(dataset, labels, 0.3);
  LogisticRegression<> lr2(dataset.n_rows, 0.3);
  lr2.Train(dataset, labels);

  REQUIRE(lr.Parameters().n_elem == lr2.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    REQUIRE(lr.Parameters()[i] == Approx(lr2.Parameters()[i]).epsilon(0.00005));
}

/**
 * Test the Train() function and make sure it works the same as if we'd called
 * the constructor by hand, with the SGD optimizer.
 */
TEST_CASE("LogisticRegressionSGDTrainTest", "[LogisticRegressionTest]")
{
  // Make a random dataset with random labels.
  arma::mat dataset(5, 800);
  dataset.randu();
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  ens::SGD<> sgd;
  sgd.Shuffle() = false;
  LogisticRegression<> lr(dataset, labels, sgd, 0.3);

  ens::SGD<> sgd2;
  sgd2.Shuffle() = false;
  LogisticRegression<> lr2(dataset.n_rows, 0.3);
  lr2.Train(dataset, labels, sgd2);

  REQUIRE(lr.Parameters().n_elem == lr2.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    REQUIRE(lr.Parameters()[i] == Approx(lr2.Parameters()[i]).epsilon(1e-7));
}

/**
 * Test sparse and dense logistic regression and make sure they both work the
 * same using the L-BFGS optimizer.
 */
TEST_CASE("LogisticRegressionSparseLBFGSTest", "[LogisticRegressionTest]")
{
  // Create a random dataset.
  arma::sp_mat dataset;
  dataset.sprandu(10, 800, 0.3);
  arma::mat denseDataset(dataset);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LogisticRegression<> lr(denseDataset, labels, 0.3);
  LogisticRegression<arma::sp_mat> lrSparse(dataset, labels, 0.3);

  REQUIRE(lr.Parameters().n_elem == lrSparse.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    REQUIRE(lr.Parameters()[i] ==
        Approx(lrSparse.Parameters()[i]).epsilon(1e-2));
}

/**
 * Test sparse and dense logistic regression and make sure they both work the
 * same using the SGD optimizer.
 */
TEST_CASE("LogisticRegressionSparseSGDTest", "[LogisticRegressionTest]")
{
  // Create a random dataset.
  arma::sp_mat dataset;
  dataset.sprandu(10, 800, 0.3);
  arma::mat denseDataset(dataset);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LogisticRegression<> lr(10, 0.3);
  ens::SGD<> sgd;
  sgd.Shuffle() = false;
  lr.Train(denseDataset, labels, sgd);

  LogisticRegression<arma::sp_mat> lrSparse(10, 0.3);
  ens::SGD<> sgdSparse;
  sgdSparse.Shuffle() = false;
  lrSparse.Train(dataset, labels, sgdSparse);

  REQUIRE(lr.Parameters().n_elem == lrSparse.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    REQUIRE(lr.Parameters()[i] ==
        Approx(lrSparse.Parameters()[i]).epsilon(1e-5));
}

/**
 * Test multi-point classification (Classify()).
 */
TEST_CASE("ClassifyTest", "[LogisticRegressionTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  lr.Train<>(data, responses);

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  arma::Row<size_t> predictions;
  lr.Classify(data, predictions);

  REQUIRE((double) arma::accu(predictions == responses) >= 900);
}

/**
 * Test that single-point classification gives the same results as multi-point
 * classification.
 */
TEST_CASE("LogisticRegressionSinglePointClassifyTest",
          "[LogisticRegressionTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  lr.Train<>(data, responses);

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  arma::Row<size_t> predictions;
  lr.Classify(data, predictions);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t pred = lr.Classify(data.col(i));

    REQUIRE(pred == predictions[i]);
  }
}

/**
 * Test that giving point probabilities works.
 */
TEST_CASE("ClassifyProbabilitiesTest", "[LogisticRegressionTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  lr.Train<>(data, responses);

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  arma::mat probabilities;
  lr.Classify(data, probabilities);

  REQUIRE(probabilities.n_cols == data.n_cols);
  REQUIRE(probabilities.n_rows == 2);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(probabilities(0, i) + probabilities(1, i) ==
        Approx(1.0).epsilon(1e-7));

    // 10% tolerance.
    if (responses[i] == 0)
      REQUIRE(probabilities(0, i) == Approx(1.0).epsilon(0.10));
    else
      REQUIRE(probabilities(1, i) == Approx(1.0).epsilon(0.10));
  }
}

/**
 * Test that LogisticRegression::Train() returns finite final objective
 * value.
 */
TEST_CASE("LogisticRegressionTrainReturnObjective", "[LogisticRegressionTest]")
{
  // Very simple fake dataset.
  arma::mat data("1 2 3;"
                 "1 2 3");
  arma::Row<size_t> responses("1 1 0");

  // Check with L_BFGS optimizer.
  LogisticRegression<> lr1(data.n_rows, 0.5);
  double objVal = lr1.Train<ens::L_BFGS>(data, responses);

  REQUIRE(std::isfinite(objVal) == true);

  // Check with a pre-defined L_BFGS optimizer.
  LogisticRegression<> lr2(data.n_rows, 0.5);
  ens::L_BFGS lbfgsOpt;
  objVal = lr2.Train(data, responses, lbfgsOpt);

  REQUIRE(std::isfinite(objVal) == true);

  // Check with SGD optimizer.
  LogisticRegression<> lr3(data.n_rows, 0.5);
  objVal = lr3.Train<ens::StandardSGD>(data, responses);

  REQUIRE(std::isfinite(objVal) == true);

  // Check with pre-defined SGD optimizer.
  LogisticRegression<> lr4(data.n_rows, 0.0005);
  ens::StandardSGD sgdOpt;
  sgdOpt.StepSize() = 0.15;
  sgdOpt.Tolerance() = 1e-75;
  objVal = lr4.Train(data, responses, sgdOpt);

  REQUIRE(std::isfinite(objVal) == true);
}

/**
 * Test that construction *then* training works fine.  Thanks @Trento89 for the
 * test case (see #2358).
 */
TEST_CASE("ConstructionThenTraining", "[LogisticRegressionTest]")
{
  arma::mat myMatrix;

  // Four points, three dimensions.
  myMatrix = { { 0.555950, 0.274690, 0.540605, 0.798938 },
               { 0.948014, 0.973234, 0.216504, 0.883152 },
               { 0.023787, 0.675382, 0.231751, 0.450332 } };

  arma::Row<size_t> myTargets("1 0 1 0");

  LogisticRegression<> lr;

  // Make sure that training doesn't crash with invalid parameter sizes.
  REQUIRE_NOTHROW(lr.Train(myMatrix, myTargets));
}

/**
 * Make sure that incremental training works.
 */
TEST_CASE("IncrementalTraining", "[LogisticRegressionTest]")
{
  // Generate a two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  LogisticRegression<> lr(data.n_rows, 0.5);
  for (size_t epoch = 0; epoch < 10; ++epoch)
    for (size_t i = 0; i < data.n_cols; ++i)
      lr.Train<ens::StandardSGD>(data, responses);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);

  REQUIRE(acc == Approx(100.0).epsilon(0.03)); // 3% error tolerance.
}
