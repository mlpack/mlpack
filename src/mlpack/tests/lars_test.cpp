/**
 * @file tests/lars_test.cpp
 * @author Nishant Mehta
 *
 * Test for LARS.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/lars.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace mlpack;

template<typename MatType, typename ResponsesType>
void GenerateProblem(
    MatType& X, ResponsesType& y, size_t nPoints, size_t nDims)
{
  X = arma::randn<MatType>(nDims, nPoints);
  arma::Col<typename ResponsesType::elem_type> beta =
      arma::randn<arma::Col<typename ResponsesType::elem_type>>(nDims);
  y = beta.t() * X;
}

template<typename VecType, typename ElemType>
void LARSVerifyCorrectness(const VecType& beta,
                           const VecType& errCorr,
                           ElemType lambda)
{
  size_t nDims = beta.n_elem;

  // floats require a much larger tolerance.
  const ElemType tol = (std::is_same_v<ElemType, double>) ? 1e-8 : 5e-3;

  for (size_t j = 0; j < nDims; ++j)
  {
    if (beta(j) == 0)
    {
      // Make sure that |errCorr(j)| <= lambda.
      REQUIRE(std::max(fabs(errCorr(j)) - lambda, (ElemType) 0.0) ==
          Approx(0.0).margin(tol));
    }
    else if (beta(j) < 0)
    {
      // Make sure that errCorr(j) == lambda.
      REQUIRE(errCorr(j) - lambda == Approx(0.0).margin(tol));
    }
    else // beta(j) > 0
    {
      // Make sure that errCorr(j) == -lambda.
      REQUIRE(errCorr(j) + lambda == Approx(0.0).margin(tol));
    }
  }
}

template<typename MatType>
void LassoTest(size_t nPoints, size_t nDims, bool elasticNet, bool useCholesky,
               bool fitIntercept, bool normalizeData)
{
  using ElemType = typename MatType::elem_type;

  MatType X;
  arma::Row<ElemType> y;

  for (size_t i = 0; i < 100; ++i)
  {
    GenerateProblem(X, y, nPoints, nDims);

    // Armadillo's median is broken, so...
    arma::Col<ElemType> sortedAbsCorr = sort(abs(X * y.t()));
    ElemType lambda1 = sortedAbsCorr(nDims / 2);
    ElemType lambda2;
    if (elasticNet)
      lambda2 = lambda1 / 2;
    else
      lambda2 = 0;

    LARS<MatType> lars(useCholesky, lambda1, lambda2);
    lars.FitIntercept(fitIntercept);
    lars.NormalizeData(normalizeData);
    lars.Train(X, y);
    arma::Col<ElemType> betaOpt = lars.Beta();

    if (fitIntercept)
    {
      y -= arma::mean(y);
      X.each_col() -= arma::mean(X, 1);
    }

    if (normalizeData)
    {
      arma::Col<ElemType> stds = arma::stddev(X, 0, 1);
      stds.replace(0.0, 1.0);

      X.each_col() /= stds;
      betaOpt %= stds; // recover solution in normalized space
    }

    arma::Col<ElemType> errCorr = (X * trans(X) + lambda2 *
        arma::eye<MatType>(nDims, nDims)) * betaOpt - X * y.t();

    LARSVerifyCorrectness(betaOpt, errCorr, lambda1);
  }
}

TEMPLATE_TEST_CASE("LARSTestLassoCholesky", "[LARSTest]", arma::fmat, arma::mat)
{
  LassoTest<TestType>(100, 10, false, true, false, false);
  LassoTest<TestType>(100, 10, false, true, true, false);
  LassoTest<TestType>(100, 10, false, true, false, true);
  LassoTest<TestType>(100, 10, false, true, true, true);
}


TEMPLATE_TEST_CASE("LARSTestLassoGram", "[LARSTest]", arma::fmat, arma::mat)
{
  LassoTest<TestType>(100, 10, false, false, false, false);
  LassoTest<TestType>(100, 10, false, false, true, false);
  LassoTest<TestType>(100, 10, false, false, false, true);
  LassoTest<TestType>(100, 10, false, false, true, true);
}

TEMPLATE_TEST_CASE("LARSTestElasticNetCholesky", "[LARSTest]", arma::fmat,
    arma::mat)
{
  LassoTest<TestType>(100, 10, true, true, false, false);
  LassoTest<TestType>(100, 10, true, true, true, false);
  LassoTest<TestType>(100, 10, true, true, false, true);
  LassoTest<TestType>(100, 10, true, true, true, true);
}

TEMPLATE_TEST_CASE("LARSTestElasticNetGram", "[LARSTest]", arma::fmat,
    arma::mat)
{
  LassoTest<TestType>(100, 10, true, false, false, false);
  LassoTest<TestType>(100, 10, true, false, true, false);
  LassoTest<TestType>(100, 10, true, false, false, true);
  LassoTest<TestType>(100, 10, true, false, true, true);
}

// Ensure that LARS doesn't crash when the data has linearly dependent features
// (meaning that there is a singularity).  This test uses the Cholesky
// factorization.
TEST_CASE("CholeskySingularityTest", "[LARSTest]")
{
  arma::mat X;
  arma::mat Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::rowvec y = Y.row(0);

  // Test for a couple values of lambda1.
  for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.1)
  {
    LARS<> lars(true, lambda1, 0.0);
    lars.FitIntercept(false);
    lars.NormalizeData(false);
    lars.Train(X, y);

    arma::vec errCorr = (X * X.t()) * lars.Beta() - X * y.t();

    LARSVerifyCorrectness(lars.Beta(), errCorr, lambda1);
  }
}

// Same as the above test but with no cholesky factorization.
TEST_CASE("NoCholeskySingularityTest", "[LARSTest]")
{
  arma::mat X;
  arma::mat Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::rowvec y = Y.row(0);

  // Test for a couple values of lambda1.
  for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.1)
  {
    LARS<> lars(false, lambda1, 0.0);
    lars.FitIntercept(false);
    lars.NormalizeData(false);
    lars.Train(X, y);

    arma::vec errCorr = (X * X.t()) * lars.Beta() - X * y.t();

    LARSVerifyCorrectness(lars.Beta(), errCorr, lambda1);
  }
}

// Make sure that Predict() provides reasonable enough solutions.
TEST_CASE("PredictTest", "[LARSTest]")
{
  for (size_t i = 0; i < 2; ++i)
  {
    // Run with both true and false.
    bool useCholesky = bool(i);

    arma::mat X;
    arma::rowvec y;

    GenerateProblem(X, y, 1000, 100);

    for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.2)
    {
      for (double lambda2 = 0.0; lambda2 < 1.0; lambda2 += 0.2)
      {
        LARS<> lars(useCholesky, lambda1, lambda2);
        lars.FitIntercept(false);
        lars.NormalizeData(false);
        lars.Train(X, y);

        // Calculate what the actual error should be with these regression
        // parameters.
        arma::vec betaOptPred = (X * X.t()) * lars.Beta();
        arma::rowvec predictions;
        lars.Predict(X, predictions);
        arma::vec adjPred = X * predictions.t();

        const double tol = 1e-7;

        REQUIRE(predictions.n_elem == 1000);
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            REQUIRE(adjPred[i] == Approx(0.0).margin(1e-5));
          else
            REQUIRE(adjPred[i] == Approx(betaOptPred[i]).epsilon(tol));
        }

        // Now check with single-point Predict(), in two ways: we will pass
        // different types into Predict() to test templating support.  We allow
        // a looser tolerance for predictions.
        for (size_t i = 0; i < X.n_cols; ++i)
          predictions[i] = lars.Predict(X.col(i));

        adjPred = X * predictions.t();
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            REQUIRE(adjPred[i] == Approx(0.0).margin(1e-5));
          else
            REQUIRE(adjPred[i] == Approx(betaOptPred[i]).epsilon(tol));
        }

        for (size_t i = 0; i < X.n_cols; ++i)
          predictions[i] = lars.Predict(X.unsafe_col(i));

        adjPred = X * predictions.t();
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            REQUIRE(adjPred[i] == Approx(0.0).margin(1e-5));
          else
            REQUIRE(adjPred[i] == Approx(betaOptPred[i]).epsilon(tol));
        }
      }
    }
  }
}

// This is the same as PredictTest, but for arma::fmat, and it allows multiple
// trials for run to deal with the lower precision of floats.
TEST_CASE("PredictFloatTest", "[LARSTest]")
{
  for (size_t i = 0; i < 2; ++i)
  {
    // Run with both true and false.
    bool useCholesky = bool(i);

    arma::fmat X;
    arma::frowvec y;

    for (float lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.2)
    {
      for (float lambda2 = 0.0; lambda2 < 1.0; lambda2 += 0.2)
      {
        // For float data, sometimes the solutions are further away from the
        // true solution due to precision issues, so we allow multiple trials.
        bool success = false;
        for (size_t trial = 0; trial < 3; ++trial)
        {
          // Generate a new problem so that we hopefully end up with a better
          // fit.
          GenerateProblem(X, y, 1000, 100);

          LARS<arma::fmat> lars(useCholesky, lambda1, lambda2);
          lars.FitIntercept(false);
          lars.NormalizeData(false);
          lars.Train(X, y);

          // Calculate what the actual error should be with these regression
          // parameters.
          arma::fvec betaOptPred = (X * X.t()) * lars.Beta();
          arma::frowvec predictions;
          lars.Predict(X, predictions);
          arma::fvec adjPred = X * predictions.t();

          const float tol = 3e-5;

          REQUIRE(predictions.n_elem == 1000);
          bool trialSuccess = true;
          for (size_t i = 0; i < betaOptPred.n_elem; ++i)
          {
            if (std::abs(betaOptPred[i]) < 1e-5)
            {
              if (adjPred[i] != Approx(0.0).margin(1e-5))
              {
                trialSuccess = false;
                break;
              }
            }
            else
            {
              if (adjPred[i] != Approx(betaOptPred[i]).epsilon(tol))
              {
                trialSuccess = false;
                break;
              }
            }
          }

          // If this trial didn't succeed, skip to the next trial.
          if (!trialSuccess)
            continue;

          // Now check with single-point Predict(), in two ways: we will pass
          // different types into Predict() to test templating support.  We
          // allow a looser tolerance for predictions.
          for (size_t i = 0; i < X.n_cols; ++i)
            predictions[i] = lars.Predict(X.col(i));

          adjPred = X * predictions.t();
          for (size_t i = 0; i < betaOptPred.n_elem; ++i)
          {
            if (std::abs(betaOptPred[i]) < 1e-5)
            {
              if (adjPred[i] != Approx(0.0).margin(1e-5))
              {
                trialSuccess = false;
                break;
              }
            }
            else
            {
              if (adjPred[i] != Approx(betaOptPred[i]).epsilon(10 * tol))
              {
                trialSuccess = false;
                break;
              }
            }
          }

          // If this trial didn't succeed, skip to the next trial.
          if (!trialSuccess)
            continue;

          for (size_t i = 0; i < X.n_cols; ++i)
            predictions[i] = lars.Predict(X.unsafe_col(i));

          adjPred = X * predictions.t();
          for (size_t i = 0; i < betaOptPred.n_elem; ++i)
          {
            if (std::abs(betaOptPred[i]) < 1e-5)
            {
              if (adjPred[i] != Approx(0.0).margin(1e-5))
              {
                trialSuccess = false;
                break;
              }
            }
            else
            {
              if (adjPred[i] != Approx(betaOptPred[i]).epsilon(10 * tol))
              {
                trialSuccess = false;
                break;
              }
            }
          }

          // If this trial succeeded, we're done.
          if (trialSuccess)
          {
            success = true;
            break;
          }
        }

        REQUIRE(success == true);
      }
    }
  }
}

TEST_CASE("PredictRowMajorTest", "[LARSTest]")
{
  arma::mat X;
  arma::rowvec y;
  GenerateProblem(X, y, 1000, 100);

  // Set lambdas to 0.

  LARS<> lars(false, 0, 0);
  lars.FitIntercept(false);
  lars.NormalizeData(false);
  lars.Train(X, y);

  // Get both row-major and column-major predictions.  Make sure they are the
  // same.
  arma::rowvec rowMajorPred, colMajorPred;

  lars.Predict(X, colMajorPred);
  lars.Predict(X.t(), rowMajorPred, false);

  REQUIRE(colMajorPred.n_elem == rowMajorPred.n_elem);
  for (size_t i = 0; i < colMajorPred.n_elem; ++i)
  {
    if (std::abs(colMajorPred[i]) < 1e-5)
      REQUIRE(rowMajorPred[i] == Approx(0.0).margin(1e-5));
    else
      REQUIRE(colMajorPred[i] == Approx(rowMajorPred[i]).epsilon(1e-7));
  }
}

/**
 * Make sure that if we train twice, there is no issue.
 */
TEST_CASE("LARSRetrainTest", "[LARSTest]")
{
  arma::mat origX;
  arma::rowvec origY;
  GenerateProblem(origX, origY, 1000, 50);

  arma::mat newX;
  arma::rowvec newY;
  GenerateProblem(newX, newY, 750, 75);

  LARS<> lars(false, 0.1, 0.1);
  lars.FitIntercept(false);
  lars.NormalizeData(false);
  lars.Train(origX, origY);

  // Now train on new data.
  lars.Train(newX, newY);

  arma::vec errCorr = (newX * trans(newX) + 0.1 *
        arma::eye(75, 75)) * lars.Beta() - newX * newY.t();

  LARSVerifyCorrectness(lars.Beta(), errCorr, 0.1);
}

/**
 * Make sure if we train twice using the Cholesky decomposition, there is no
 * issue.
 */
TEST_CASE("RetrainCholeskyTest", "[LARSTest]")
{
  arma::mat origX;
  arma::rowvec origY;
  GenerateProblem(origX, origY, 1000, 50);

  arma::mat newX;
  arma::rowvec newY;
  GenerateProblem(newX, newY, 750, 75);

  LARS<> lars(true, 0.1, 0.1);
  lars.FitIntercept(false);
  lars.NormalizeData(false);
  lars.Train(origX, origY);

  // Now train on new data.
  lars.Train(newX, newY);

  arma::vec errCorr = (newX * trans(newX) + 0.1 *
        arma::eye(75, 75)) * lars.Beta() - newX * newY.t();

  LARSVerifyCorrectness(lars.Beta(), errCorr, 0.1);
}

/**
 * Make sure that we get correct solution coefficients when running training
 * and accessing solution coefficients separately.
 */
TEST_CASE("TrainingAndAccessingBetaTest", "[LARSTest]")
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 1000, 100);

  LARS<> lars1;
  lars1.Train(X, y);
  arma::vec beta = lars1.Beta();

  LARS<> lars2;
  lars2.Train(X, y);

  REQUIRE(beta.n_elem == lars2.Beta().n_elem);
  for (size_t i = 0; i < beta.n_elem; ++i)
    REQUIRE(beta[i] == Approx(lars2.Beta()[i]).epsilon(1e-7));
}

/**
 * Make sure that we learn the same when running training separately and through
 * constructor. Test it with default parameters.
 */
TEST_CASE("TrainingConstructorWithDefaultsTest", "[LARSTest]")
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 1000, 100);

  LARS<> lars1;
  lars1.Train(X, y);
  arma::vec beta = lars1.Beta();

  LARS<> lars2(X, y);

  REQUIRE(beta.n_elem == lars2.Beta().n_elem);
  for (size_t i = 0; i < beta.n_elem; ++i)
    REQUIRE(beta[i] == Approx(lars2.Beta()[i]).epsilon(1e-7));
}

/**
 * Make sure that we learn the same when running training separately and through
 * constructor. Test it with non default parameters.
 */
TEST_CASE("TrainingConstructorWithNonDefaultsTest", "[LARSTest]")
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 1000, 100);

  bool transposeData = true;
  bool useCholesky = true;
  double lambda1 = 0.2;
  double lambda2 = 0.4;

  LARS<> lars1(useCholesky, lambda1, lambda2);
  lars1.Train(X, y);
  arma::vec beta = lars1.Beta();

  LARS<> lars2(X, y, transposeData, useCholesky, lambda1, lambda2);

  REQUIRE(beta.n_elem == lars2.Beta().n_elem);
  for (size_t i = 0; i < beta.n_elem; ++i)
    REQUIRE(beta[i] == Approx(lars2.Beta()[i]).epsilon(1e-7));
}

/**
 * Test that LARS::Train() returns finite error value.
 */
TEST_CASE("LARSTrainReturnCorrelation", "[LARSTest]")
{
  arma::mat X;
  arma::mat Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::rowvec y = Y.row(0);

  double lambda1 = 0.1;
  double lambda2 = 0.1;

  // Test with Cholesky decomposition and with lasso.
  LARS<> lars1(true, lambda1, 0.0);
  double error = lars1.Train(X, y);

  REQUIRE(std::isfinite(error) == true);

  // Test without Cholesky decomposition and with lasso.
  LARS<> lars2(false, lambda1, 0.0);
  error = lars2.Train(X, y);

  REQUIRE(std::isfinite(error) == true);

  // Test with Cholesky decomposition and with elasticnet.
  LARS<> lars3(true, lambda1, lambda2);
  error = lars3.Train(X, y);

  REQUIRE(std::isfinite(error) == true);

  // Test without Cholesky decomposition and with elasticnet.
  LARS<> lars4(false, lambda1, lambda2);
  error = lars4.Train(X, y);

  REQUIRE(std::isfinite(error) == true);
}

/**
 * Test that LARS::ComputeError() returns error value less than 1
 * and greater than 0.
 */
TEMPLATE_TEST_CASE("LARSTestComputeError", "[LARSTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;

  MatType X;
  MatType Y;

  if (!data::Load("lars_dependent_x.csv", X))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");

  arma::Row<ElemType> y = Y.row(0);

  LARS<MatType> lars1(true, 0.1, 0.0);
  lars1.FitIntercept(false);
  lars1.NormalizeData(false);
  ElemType train1 = lars1.Train(X, y);
  ElemType cost = lars1.ComputeError(X, y);

  REQUIRE(cost <= 1);
  REQUIRE(cost >= 0);
  REQUIRE(cost == train1);
}

/**
 * Simple test for LARS copy constructor.
 */
TEST_CASE("LARSCopyConstructorTest", "[LARSTest]")
{
  arma::mat features, Y;
  arma::rowvec targets;

  // Load training input and predictions for testing.
  if (!data::Load("lars_dependent_x.csv", features))
    FAIL("Cannot load dataset lars_dependent_x.csv");
  if (!data::Load("lars_dependent_y.csv", Y))
    FAIL("Cannot load dataset lars_dependent_y.csv");
  targets = Y.row(0);

  // Check if the copy is accessible even after deleting the pointer to the
  // object.
  LARS<>* glm1 = new LARS<>(false, .1, .1);
  arma::rowvec predictions, predictionsFromCopiedModel;
  std::vector<LARS<>> models;
  glm1->Train(features, targets);
  glm1->Predict(features, predictions);
  models.emplace_back(*glm1); // Call the copy constructor.
  delete glm1; // Free LARS internal memory.
  models[0].Predict(features, predictionsFromCopiedModel);
  // The output of both models should be the same.
  CheckMatrices(predictions, predictionsFromCopiedModel);
  // Check if we can train the model again.
  REQUIRE_NOTHROW(models[0].Train(features, targets));

  // Check if we can train the copied model.
  LARS<> glm2(false, 0.1, 0.1);
  models.emplace_back(glm2); // Call the copy constructor.
  REQUIRE_NOTHROW(glm2.Train(features, targets));
  REQUIRE_NOTHROW(models[1].Train(features, targets));

  // Create a copy using assignment operator.
  LARS<> glm3 = glm2;
  models[1].Predict(features, predictions);
  glm3.Predict(features, predictionsFromCopiedModel);
  // The output of both models should be the same.
  CheckMatrices(predictions, predictionsFromCopiedModel);
}

/**
 * Test that fitting an intercept is the same as scaling data.
 */
TEST_CASE("LARSFitInterceptTest", "[LARSTest]")
{
  arma::mat features = arma::randu<arma::mat>(10, 100);
  arma::rowvec responses = arma::randu<arma::rowvec>(100);

  arma::mat centeredFeatures = features.each_col() - arma::mean(features, 1);
  arma::rowvec centeredResponses = responses - arma::mean(responses);

  LARS<> l1(features, responses, true, true, 0.001, 0.001, 1e-16, true, false);
  LARS<> l2(centeredFeatures, centeredResponses, true, true, 0.001, 0.001,
      1e-16, false, false);

  // The weights learned should be the same.
  REQUIRE(l1.Beta().n_elem == l2.Beta().n_elem);
  CheckMatrices(l1.Beta(), l2.Beta());
  REQUIRE(l1.Intercept() == Approx(arma::mean(responses) -
      dot(arma::mean(features, 1), l1.Beta())));
}

// Make sure that Predict() provides reasonable enough solutions when we are
// fitting an intercept.
TEST_CASE("PredictFitInterceptTest", "[LARSTest]")
{
  for (size_t i = 0; i < 2; ++i)
  {
    // Run with both true and false.
    bool useCholesky = bool(i);

    arma::mat X;
    arma::rowvec y;

    GenerateProblem(X, y, 1000, 100);

    for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.2)
    {
      for (double lambda2 = 0.0; lambda2 < 1.0; lambda2 += 0.2)
      {
        LARS<> lars(useCholesky, lambda1, lambda2);
        lars.FitIntercept(true);
        lars.NormalizeData(false);
        lars.Train(X, y);
        const double intercept = arma::mean(y) -
            dot(arma::mean(X, 1), lars.Beta());

        // Calculate what the actual error should be with these regression
        // parameters.
        arma::vec betaOptPred = X.t() * lars.Beta() + intercept;
        arma::rowvec predictions;
        lars.Predict(X, predictions);
        arma::vec adjPred = predictions.t();

        REQUIRE(predictions.n_elem == 1000);
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            REQUIRE(adjPred[i] == Approx(0.0).margin(1e-5));
          else
            REQUIRE(adjPred[i] == Approx(betaOptPred[i]).epsilon(1e-7));
        }
      }
    }
  }
}

// Make sure that Predict() provides reasonable enough solutions when we are
// normalizing data.
TEST_CASE("PredictNormalizeDataTest", "[LARSTest]")
{
  for (size_t i = 0; i < 2; ++i)
  {
    // Run with both true and false.
    bool useCholesky = bool(i);

    arma::mat X;
    arma::rowvec y;

    GenerateProblem(X, y, 1000, 100);

    for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.2)
    {
      for (double lambda2 = 0.0; lambda2 < 1.0; lambda2 += 0.2)
      {
        LARS<> lars(useCholesky, lambda1, lambda2);
        lars.FitIntercept(false);
        lars.NormalizeData(true);
        lars.Train(X, y);

        // Calculate what the actual error should be with these regression
        // parameters.
        arma::vec betaOptPred = (X * X.t()) * lars.Beta();
        arma::rowvec predictions;
        lars.Predict(X, predictions);
        arma::vec adjPred = X * predictions.t();

        REQUIRE(predictions.n_elem == 1000);
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            REQUIRE(adjPred[i] == Approx(0.0).margin(1e-5));
          else
            REQUIRE(adjPred[i] == Approx(betaOptPred[i]).epsilon(1e-7));
        }
      }
    }
  }
}

// Make sure that Predict() provides reasonable enough solutions when we are
// fitting an intercept and normalizing data.
TEST_CASE("PredictFitInterceptNormalizeDataTest", "[LARSTest]")
{
  for (size_t i = 0; i < 2; ++i)
  {
    // Run with both true and false.
    bool useCholesky = bool(i);

    arma::mat X;
    arma::rowvec y;

    GenerateProblem(X, y, 1000, 100);

    for (double lambda1 = 0.0; lambda1 < 1.0; lambda1 += 0.2)
    {
      for (double lambda2 = 0.0; lambda2 < 1.0; lambda2 += 0.2)
      {
        LARS<> lars(useCholesky, lambda1, lambda2);
        lars.FitIntercept(true);
        lars.NormalizeData(true);
        lars.Train(X, y);
        const double intercept = arma::mean(y) -
            dot(arma::mean(X, 1), lars.Beta());

        // Calculate what the actual error should be with these regression
        // parameters.
        arma::vec betaOptPred = X.t() * lars.Beta() + intercept;
        arma::rowvec predictions;
        lars.Predict(X, predictions);
        arma::vec adjPred = predictions.t();

        REQUIRE(predictions.n_elem == 1000);
        for (size_t i = 0; i < betaOptPred.n_elem; ++i)
        {
          if (std::abs(betaOptPred[i]) < 1e-5)
            REQUIRE(adjPred[i] == Approx(0.0).margin(1e-5));
          else
            REQUIRE(adjPred[i] == Approx(betaOptPred[i]).epsilon(1e-7));
        }
      }
    }
  }
}

/**
 * Verify that KKT conditions are satisfied for a solution of the standard Lasso
 * problem (with X, y and lambda) (own version)
 *
 * A solution \bar{beta} of the standard Lasso problem verifies
 *
 * \bar{beta} = argmin 1/2* || X.beta-y||^2 + lambda*|beta|_1
 *
 * @param beta vector of double values, computed solution of the standard Lasso problem
 * @param X matrix of the standard Lasso problem
 * @param y vector of doubles, from the standard Lasso problem, as an armadillo rowvec
 * @param lambda double, parameter in the standard Lasso problem
 */
void CheckKKT(const arma::vec& beta,
              const arma::mat& X,
              const arma::rowvec& y,
              const double lambda)
{
  const double epsilon = 1e-6; // For numerical precision.

  arma::vec v = X.t() * X * beta - X.t() * y.t() + lambda * sign(beta);
  // Active set indices with global numbering: could be empty.
  arma::uvec ia = arma::find(arma::abs(beta) > epsilon);
  // Zero indices with global numbering: could be empty.
  arma::uvec iz = arma::find(arma::abs(beta) <= epsilon);

  // Should be zero if beta is the solution.
  const double crit = dot(beta, v);
  REQUIRE(std::abs(crit) < epsilon);

  // v should be zero at the Active Set ia.
  REQUIRE(arma::all(arma::abs(v(ia)) <= epsilon));

  // We should have abs(v) <= lambda at zero values Iz
  REQUIRE(arma::all(v(iz) >= -lambda));
  REQUIRE(arma::all(v(iz) <= lambda));
}

TEST_CASE("LARSTestKKT", "[LARSTest]")
{
  // Each row of F corresponds to a test.
  //
  // For each test i,
  //    F(0, i) is the matrix of covariates X, and
  //    F(1, i) is the matrix (vector) of responses/observations y.
  arma::field<arma::mat> F;
  F.load("lars_kkt.bin");

  bool useCholesky = true;
  LARS<> lars(useCholesky, 1.0, 0.0);

  for (size_t i = 0; i < F.n_cols; i++)
  {
    arma::mat X = std::move(F(0, i));
    arma::rowvec y = std::move(F(1, i));
    const arma::rowvec xMean = arma::mean(X, 0);
    arma::rowvec xStds = arma::stddev(X, 0, 0);
    xStds.replace(0.0, 1.0);
    const double yMean = arma::mean(y);

    lars.FitIntercept(false);
    lars.NormalizeData(false);
    lars.Train(X, y, false);
    CheckKKT(lars.Beta(), X, y, 1.0);

    // Now try when we fit an intercept too.
    lars.FitIntercept(true);
    lars.NormalizeData(false);
    lars.Train(X, y, false);

    // Now mean-center data before the check.
    X.each_row() -= xMean;
    y -= yMean;

    CheckKKT(lars.Beta(), X, y, 1.0);

    X.each_row() += xMean;
    y += yMean;

    // Now try when we normalize the data.
    lars.FitIntercept(false);
    lars.NormalizeData(true);
    lars.Train(X, y, false);

    X.each_row() /= xStds;
    arma::vec beta = lars.Beta();
    beta %= xStds.t();

    CheckKKT(beta, X, y, 1.0);

    X.each_row() %= xStds;

    lars.FitIntercept(true);
    lars.NormalizeData(true);
    lars.Train(X, y, false);
    beta = lars.Beta();

    X.each_row() -= xMean;
    X.each_row() /= xStds;
    beta %= xStds.t();
    y -= yMean;

    CheckKKT(beta, X, y, 1.0);
  }
}

// Check that all variants of constructors appear to work.
TEMPLATE_TEST_CASE("LARSConstructorVariantTest", "[LARSTest]", arma::fmat,
    arma::mat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;

  // The results of the training are not all that important here; the more
  // important thing is just that all the overloads compile properly.  We do
  // some basic sanity checks on the trained model nonetheless.
  MatType X;
  arma::Row<ElemType> y;

  GenerateProblem(X, y, 1000, 100);
  MatType Xt = X.t();

  const arma::Col<ElemType> xMean = arma::mean(X, 1);
  arma::Col<ElemType> xStds = arma::stddev(X, 0, 1);
  xStds.replace(0.0, 1.0);
  const ElemType yMean = arma::mean(y);

  MatType centeredX = X.each_col() - xMean;
  arma::Row<ElemType> centeredY = y - yMean;

  MatType centeredUnitX = centeredX.each_col() / xStds;

  MatType matGram = X * X.t();
  MatType centeredUnitMatGram = centeredUnitX * centeredUnitX.t();

  LARS<MatType> l1;
  LARS<MatType> l2(false, 0.1, 0.2, 1e-15, false, false);
  LARS<MatType> l3(X, y);
  LARS<MatType> l4(Xt, y, false);
  LARS<MatType> l5(Xt, y, false, false);
  LARS<MatType> l6(Xt, y, false, false, 0.1);
  LARS<MatType> l7(X, y, true, false, 0.11, 0.01);
  LARS<MatType> l8(X, y, true, false, 0.12, 0.02, 1e-8);
  LARS<MatType> l9(centeredX, centeredY, true, false, 0.13, 0.03, 1e-7, false);
  LARS<MatType> l10(centeredUnitX, centeredY, true, false, 0.14, 0.04, 1e-6,
      false, false);

  REQUIRE(l1.BetaPath().size() == 0);

  REQUIRE(l2.BetaPath().size() == 0);
  REQUIRE(l2.UseCholesky() == false);
  REQUIRE(l2.Lambda1() == Approx(0.1));
  REQUIRE(l2.Lambda2() == Approx(0.2));
  REQUIRE(l2.Tolerance() == Approx(1e-15));
  REQUIRE(l2.FitIntercept() == false);
  REQUIRE(l2.NormalizeData() == false);

  REQUIRE(l3.Beta().n_elem == X.n_rows);

  REQUIRE(l4.Beta().n_elem == X.n_rows);

  REQUIRE(l5.Beta().n_elem == X.n_rows);
  REQUIRE(l5.UseCholesky() == false);

  REQUIRE(l6.Beta().n_elem == X.n_rows);
  REQUIRE(l6.UseCholesky() == false);
  REQUIRE(l6.Lambda1() == Approx(0.1));

  REQUIRE(l7.Beta().n_elem == X.n_rows);
  REQUIRE(l7.UseCholesky() == false);
  REQUIRE(l7.Lambda1() == Approx(0.11));
  REQUIRE(l7.Lambda2() == Approx(0.01));

  REQUIRE(l8.Beta().n_elem == X.n_rows);
  REQUIRE(l8.UseCholesky() == false);
  REQUIRE(l8.Lambda1() == Approx(0.12));
  REQUIRE(l8.Lambda2() == Approx(0.02));
  REQUIRE(l8.Tolerance() == Approx(1e-8));

  REQUIRE(l9.Beta().n_elem == X.n_rows);
  REQUIRE(l9.UseCholesky() == false);
  REQUIRE(l9.Lambda1() == Approx(0.13));
  REQUIRE(l9.Lambda2() == Approx(0.03));
  REQUIRE(l9.Tolerance() == Approx(1e-7));
  REQUIRE(l9.FitIntercept() == false);

  REQUIRE(l10.Beta().n_elem == X.n_rows);
  REQUIRE(l10.UseCholesky() == false);
  REQUIRE(l10.Lambda1() == Approx(0.14));
  REQUIRE(l10.Lambda2() == Approx(0.04));
  REQUIRE(l10.Tolerance() == Approx(1e-6));
  REQUIRE(l10.FitIntercept() == false);
  REQUIRE(l10.NormalizeData() == false);

  // Now check constructors where we specify the Gram matrix.

  const size_t dim = centeredUnitMatGram.n_rows;
  LARS<MatType> l11(X, y, true, false, centeredUnitMatGram);
  LARS<MatType> l12(Xt, y, false, false, centeredUnitMatGram, 0.1);

  // If lambda2 > 0, then we have to adjust the Gram matrix to account for that.
  MatType centeredUnitMatGramL13 = centeredUnitMatGram +
      0.01 * arma::eye<MatType>(dim, dim);
  LARS<MatType> l13(Xt, y, false, false, centeredUnitMatGramL13, 0.11, 0.01);

  MatType centeredUnitMatGramL14 = centeredUnitMatGram +
      0.02 * arma::eye<MatType>(dim, dim);
  LARS<MatType> l14(Xt, y, false, false, centeredUnitMatGramL14, 0.12, 0.02,
      1e-15);

  MatType centeredUnitMatGramL15 = centeredUnitMatGram +
      0.03 * arma::eye<MatType>(dim, dim);
  LARS<MatType> l15(centeredX, centeredY, true, false, centeredUnitMatGramL15,
      0.13, 0.03, 1e-14, false);

  MatType centeredUnitMatGramL16 = centeredUnitMatGram +
      0.04 * arma::eye<MatType>(dim, dim);
  LARS<MatType> l16(centeredUnitX, centeredY, true, false,
      centeredUnitMatGramL16, 0.14, 0.04, 1e-13, false, false);

  REQUIRE(l11.Beta().n_elem == X.n_rows);
  REQUIRE(l11.UseCholesky() == false);

  REQUIRE(l12.Beta().n_elem == X.n_rows);
  REQUIRE(l12.UseCholesky() == false);
  REQUIRE(l12.Lambda1() == Approx(0.1));

  REQUIRE(l13.Beta().n_elem == X.n_rows);
  REQUIRE(l13.UseCholesky() == false);
  REQUIRE(l13.Lambda1() == Approx(0.11));
  REQUIRE(l13.Lambda2() == Approx(0.01));

  REQUIRE(l14.Beta().n_elem == X.n_rows);
  REQUIRE(l14.UseCholesky() == false);
  REQUIRE(l14.Lambda1() == Approx(0.12));
  REQUIRE(l14.Lambda2() == Approx(0.02));
  REQUIRE(l14.Tolerance() == Approx(1e-15));

  REQUIRE(l15.Beta().n_elem == X.n_rows);
  REQUIRE(l15.UseCholesky() == false);
  REQUIRE(l15.Lambda1() == Approx(0.13));
  REQUIRE(l15.Lambda2() == Approx(0.03));
  REQUIRE(l15.Tolerance() == Approx(1e-14));
  REQUIRE(l15.FitIntercept() == false);

  REQUIRE(l16.Beta().n_elem == X.n_rows);
  REQUIRE(l16.UseCholesky() == false);
  REQUIRE(l16.Lambda1() == Approx(0.14));
  REQUIRE(l16.Lambda2() == Approx(0.04));
  REQUIRE(l16.Tolerance() == Approx(1e-13));
  REQUIRE(l16.FitIntercept() == false);
  REQUIRE(l16.NormalizeData() == false);
}

// Check that all variants of Train() appear to work.
TEMPLATE_TEST_CASE("LARSTrainVariantTest", "[LARSTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;

  // The results of the training are not all that important here; the more
  // important thing is just that all the overloads compile properly.  We do
  // some basic sanity checks on the trained model nonetheless.
  MatType X;
  arma::Row<ElemType> y;

  GenerateProblem(X, y, 1000, 5);
  MatType Xt = X.t();

  const arma::Col<ElemType> xMean = arma::mean(X, 1);
  arma::Col<ElemType> xStds = arma::stddev(X, 0, 1);
  xStds.replace(0.0, 1.0);
  const ElemType yMean = arma::mean(y);

  MatType centeredX = X.each_col() - xMean;
  arma::Row<ElemType> centeredY = y - yMean;

  MatType centeredUnitX = centeredX.each_col() / xStds;

  MatType matGram = X * X.t();
  MatType centeredUnitMatGram = centeredUnitX * centeredUnitX.t();

  LARS<MatType> l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14;

  l1.Train(X, y);
  l2.Train(Xt, y, false);
  l3.Train(Xt, y, false, false);
  l4.Train(Xt, y, false, false, 0.1);
  l5.Train(Xt, y, false, false, 0.11, 0.01);
  l6.Train(Xt, y, false, false, 0.12, 0.02, 1e-15);
  l7.Train(centeredX, centeredY, true, false, 0.13, 0.03, 1e-14, false);
  l8.Train(centeredX, centeredY, true, false, 0.14, 0.04, 1e-13, false, false);

  l9.Train(Xt, y, false, false, centeredUnitMatGram);
  l10.Train(X, y, true, false, centeredUnitMatGram, 0.15);

  // If lambda2 > 0, then we have to adjust the Gram matrix.
  const size_t dim = centeredUnitMatGram.n_rows;
  MatType centeredUnitMatGramL11 = centeredUnitMatGram +
      0.05 * arma::eye<MatType>(dim, dim);
  l11.Train(X, y, true, false, centeredUnitMatGramL11, 0.16, 0.05);

  MatType centeredUnitMatGramL12 = centeredUnitMatGram +
      0.06 * arma::eye<MatType>(dim, dim);
  l12.Train(X, y, true, false, centeredUnitMatGramL12, 0.17, 0.06, 1e-12);

  MatType centeredUnitMatGramL13 = centeredUnitMatGram +
      0.07 * arma::eye<MatType>(dim, dim);
  l13.Train(centeredX, centeredY, true, false, centeredUnitMatGramL13, 0.18,
      0.07, 1e-11, false);

  MatType centeredUnitMatGramL14 = centeredUnitMatGram +
      0.08 * arma::eye<MatType>(dim, dim);
  l14.Train(centeredUnitX, centeredY, true, false, centeredUnitMatGramL14, 0.19,
      0.08, 1e-10, false, false);

  REQUIRE(l1.Beta().n_elem == X.n_rows);

  REQUIRE(l2.Beta().n_elem == X.n_rows);

  REQUIRE(l3.Beta().n_elem == X.n_rows);
  REQUIRE(l3.UseCholesky() == false);

  REQUIRE(l4.Beta().n_elem == X.n_rows);
  REQUIRE(l4.UseCholesky() == false);
  REQUIRE(l4.Lambda1() == Approx(0.1));

  REQUIRE(l5.Beta().n_elem == X.n_rows);
  REQUIRE(l5.UseCholesky() == false);
  REQUIRE(l5.Lambda1() == Approx(0.11));
  REQUIRE(l5.Lambda2() == Approx(0.01));

  REQUIRE(l6.Beta().n_elem == X.n_rows);
  REQUIRE(l6.UseCholesky() == false);
  REQUIRE(l6.Lambda1() == Approx(0.12));
  REQUIRE(l6.Lambda2() == Approx(0.02));
  REQUIRE(l6.Tolerance() == Approx(1e-15));

  REQUIRE(l7.Beta().n_elem == X.n_rows);
  REQUIRE(l7.UseCholesky() == false);
  REQUIRE(l7.Lambda1() == Approx(0.13));
  REQUIRE(l7.Lambda2() == Approx(0.03));
  REQUIRE(l7.Tolerance() == Approx(1e-14));
  REQUIRE(l7.FitIntercept() == false);

  REQUIRE(l8.Beta().n_elem == X.n_rows);
  REQUIRE(l8.UseCholesky() == false);
  REQUIRE(l8.Lambda1() == Approx(0.14));
  REQUIRE(l8.Lambda2() == Approx(0.04));
  REQUIRE(l8.Tolerance() == Approx(1e-13));
  REQUIRE(l8.FitIntercept() == false);
  REQUIRE(l8.NormalizeData() == false);

  REQUIRE(l9.Beta().n_elem == X.n_rows);
  REQUIRE(l9.UseCholesky() == false);

  REQUIRE(l10.Beta().n_elem == X.n_rows);
  REQUIRE(l10.UseCholesky() == false);
  REQUIRE(l10.Lambda1() == Approx(0.15));

  REQUIRE(l11.Beta().n_elem == X.n_rows);
  REQUIRE(l11.UseCholesky() == false);
  REQUIRE(l11.Lambda1() == Approx(0.16));
  REQUIRE(l11.Lambda2() == Approx(0.05));

  REQUIRE(l12.Beta().n_elem == X.n_rows);
  REQUIRE(l12.UseCholesky() == false);
  REQUIRE(l12.Lambda1() == Approx(0.17));
  REQUIRE(l12.Lambda2() == Approx(0.06));
  REQUIRE(l12.Tolerance() == Approx(1e-12));

  REQUIRE(l13.Beta().n_elem == X.n_rows);
  REQUIRE(l13.UseCholesky() == false);
  REQUIRE(l13.Lambda1() == Approx(0.18));
  REQUIRE(l13.Lambda2() == Approx(0.07));
  REQUIRE(l13.Tolerance() == Approx(1e-11));
  REQUIRE(l13.FitIntercept() == false);

  REQUIRE(l14.Beta().n_elem == X.n_rows);
  REQUIRE(l14.UseCholesky() == false);
  REQUIRE(l14.Lambda1() == Approx(0.19));
  REQUIRE(l14.Lambda2() == Approx(0.08));
  REQUIRE(l14.Tolerance() == Approx(1e-10));
  REQUIRE(l14.FitIntercept() == false);
  REQUIRE(l14.NormalizeData() == false);
}

// Ensure that SelectBeta() works correctly.
TEMPLATE_TEST_CASE("LARSSelectBetaTest", "[LARSTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using ElemType = typename MatType::elem_type;

  const ElemType tol = (std::is_same_v<ElemType, double>) ? 1e-5 : 5e-3;

  // Train a model on a randomly generated problem.  Then, we will iterate
  // through different selected lambda values, ensuring that the error on the
  // training set is monotonically increasing.
  MatType X;
  arma::Row<ElemType> y;

  GenerateProblem(X, y, 1000, 100);

  LARS<MatType> lars(X, y);

  // Ensure that the solution with no regularization is fully dense.
  REQUIRE(lars.ActiveSet().size() == X.n_rows);
  REQUIRE(lars.Beta().n_elem == X.n_rows);

  // Now step through numerous different lambda values.
  ElemType lastError = std::numeric_limits<ElemType>::max();
  const ElemType errorTol = (std::is_same_v<ElemType, double>) ? 1e-8 : 0.05;
  for (ElemType i = 5.0; i >= -5.0; i -= 0.1)
  {
    const ElemType selLambda1 = std::pow(10.0, (ElemType) i);
    lars.SelectBeta(selLambda1);

    REQUIRE(lars.Beta().n_elem == X.n_rows);
    REQUIRE(lars.SelectedLambda1() == Approx(selLambda1).margin(tol));
    REQUIRE(accu(lars.Beta() != 0.0) == lars.ActiveSet().size());
    const ElemType newError = lars.ComputeError(X, y);
    REQUIRE(newError <= lastError + errorTol);
    lastError = newError;
  }

  // Lastly, step through values corresponding to the lambda path exactly.
  // Here we can just check that we are looking at the right model via
  // Intercept() and Beta().
  for (size_t i = 0; i < lars.LambdaPath().size(); ++i)
  {
    lars.SelectBeta(lars.LambdaPath()[i]);

    REQUIRE(lars.SelectedLambda1() == Approx(lars.LambdaPath()[i]).margin(tol));
    REQUIRE(lars.Intercept() == Approx(lars.InterceptPath()[i]).margin(tol));
    REQUIRE(arma::approx_equal(lars.Beta(), lars.BetaPath()[i], "absdiff",
        tol));
  }
}

// Test that SelectBeta() throws an error when the model is not trained.
TEST_CASE("LARSSelectBetaUntrainedModelTest", "[LARSTest]")
{
  LARS<> lars;

  REQUIRE_THROWS_AS(lars.SelectBeta(0.01), std::runtime_error);
}

// Test that SelectBeta() throws an exception when an invalid new lambda1 value
// is specified.
TEST_CASE("LARSSelectBetaInvalidLambda1Test", "[LARSTest]")
{
  arma::mat X;
  arma::rowvec y;

  GenerateProblem(X, y, 1000, 100);

  LARS<> lars;
  lars.Lambda1() = 1.0;
  lars.Train(X, y);

  REQUIRE_THROWS_AS(lars.SelectBeta(0.1), std::invalid_argument);
}

// Test that we can train a sparse model on dense data.
TEMPLATE_TEST_CASE("LARSSparseModelDenseData", "[LARSTest]", float, double)
{
  using eT = TestType;

  // 1k-dimensional data.
  arma::Mat<eT> data(1000, 500, arma::fill::randu);

  arma::SpCol<eT> betaSp;
  betaSp.sprandu(1000, 1, 0.1);
  arma::Col<eT> beta = betaSp + arma::randu<arma::Col<eT>>(1000) * 0.01;

  // Create slightly noisy responses.
  arma::Row<eT> responses = beta.t() * data +
      0.02 * arma::randu<arma::Row<eT>>(500);

  LARS<arma::SpMat<eT>> lars1(data, responses, true, true, 0.5, 0.01);
  LARS<arma::SpMat<eT>> lars2(true, 0.01, 1e-4);
  lars2.Train(data, responses);

  // Make sure we at least approximately recovered the solution vector.
  REQUIRE(lars1.Beta().n_elem == 1000);
  REQUIRE(lars2.Beta().n_elem == 1000);

  REQUIRE((arma::norm(lars1.Beta() - beta, 2) / lars1.Beta().n_elem) < 0.01);
  REQUIRE((arma::norm(lars2.Beta() - beta, 2) / lars2.Beta().n_elem) < 0.01);

  // Make some predictions and ensure they are approximately correct.
  arma::Row<eT> responses1, responses2;
  lars1.Predict(data, responses1);
  lars2.Predict(data, responses2);

  REQUIRE(responses1.n_elem == responses.n_elem);
  REQUIRE(responses2.n_elem == responses.n_elem);

  REQUIRE((accu(arma::abs(responses - responses1)) / responses.n_elem)
      < 0.1);
  REQUIRE((accu(arma::abs(responses - responses2)) / responses.n_elem)
      < 0.1);

  // Make sure ComputeError returns something reasonable.
  REQUIRE((lars1.ComputeError(data, responses) / responses.n_elem) < 1);
  REQUIRE((lars2.ComputeError(data, responses) / responses.n_elem) < 1);

  REQUIRE(lars1.ActiveSet().size() < 1000);
  REQUIRE(lars1.ActiveSet().size() > 0);

  REQUIRE(lars2.ActiveSet().size() < 1000);
  REQUIRE(lars2.ActiveSet().size() > 0);
}
