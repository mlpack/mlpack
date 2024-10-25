/**
 * @file tests/linear_svm_test.cpp
 * @author Ayush Chamoli
 *
 * Test the Linear SVM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * Callback test function, based on the EndOptimization callback function.
 */
class CallbackTestFunction
{
 public:
  CallbackTestFunction() : calledEndOptimization(false) {}

  template<typename OptimizerType, typename FunctionType, typename MatType>
  void EndOptimization(OptimizerType& /* optimizer */,
                       FunctionType& /* function */,
                       MatType& /* coordinates */)
  {
    calledEndOptimization = true;
  }

  //! Track to check if callback is executed.
  bool calledEndOptimization;
};

/**
 * A simple test for LinearSVMFunction
 */
TEST_CASE("LinearSVMFunctionEvaluate", "[LinearSVMTest]")
{
  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  LinearSVMFunction<arma::mat> svmf(dataset, labels, 2,
      0.0 /* no regularization */);

  // These were hand-calculated using Python.
  arma::mat parameters = "1 1 1 1 1;"
                         "1 1 1 1 1";
  REQUIRE(svmf.Evaluate(parameters.t()) == Approx(1.0).epsilon(1e-7));

  parameters = "2 0 1 2 2;"
               "1 2 2 2 2";
  REQUIRE(svmf.Evaluate(parameters.t()) == Approx(2.0).epsilon(1e-7));

  parameters = "-0.1425 8.3228 0.1724 -0.3374 0.1548;"
               "0.1435 0.0009 -0.1736 0.3356 -0.1544";
  REQUIRE(svmf.Evaluate(parameters.t()) == Approx(0.0).epsilon(1e-7));

  parameters = "100 3 4 5 23;"
               "43 54 67 32 64";
  REQUIRE(svmf.Evaluate(parameters.t()) == Approx(85.33333333).epsilon(1e-7));

  parameters = "3 71 22 12 6;"
               "100 39 30 57 22";
  REQUIRE(svmf.Evaluate(parameters.t()) == Approx(11.0).epsilon(1e-7));
}

/**
 * A complicated test for the LinearSVMFunction for binary-class
 * classification.
 */
TEST_CASE("LinearSVMFunctionRandomBinaryEvaluate", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 2;
  const double delta = 1.0;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters.t() * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k)
      {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + delta;
        if (margin > 0)
          hingeLoss += margin;
      }
    }
    hingeLoss /= points;

    // Compare with the value returned by the function.
    REQUIRE(svmf.Evaluate(parameters) == Approx(hingeLoss).epsilon(1e-7));
  }
}

/**
 * A complicated test for the LinearSVMFunction for multi-class
 * classification.
 */
TEST_CASE("LinearSVMFunctionRandomEvaluate", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 5;
  const double delta = 1.0;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters.t() * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k)
      {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + delta;
        if (margin > 0)
          hingeLoss += margin;
      }
    }
    hingeLoss /= points;

    // Compare with the value returned by the function.
    REQUIRE(svmf.Evaluate(parameters) == Approx(hingeLoss).epsilon(1e-7));
  }
}

/**
 * Test regularization for the LinearSVMFunction Evaluate()
 * function.
 */
TEST_CASE("LinearSVMFunctionRegularizationEvaluate", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // 3 objects for comparing regularization costs.
  LinearSVMFunction<arma::mat> svmfNoReg(data, labels, numClasses, 0);
  LinearSVMFunction<arma::mat> svmfSmallReg(data, labels, numClasses, 1);
  LinearSVMFunction<arma::mat> svmfBigReg(data, labels, numClasses, 20);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    double wL2SquaredNorm;
    wL2SquaredNorm = dot(parameters, parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 10 * wL2SquaredNorm;

    REQUIRE(svmfNoReg.Evaluate(parameters) + smallRegTerm ==
        Approx(svmfSmallReg.Evaluate(parameters)).epsilon(1e-7));
    REQUIRE(svmfNoReg.Evaluate(parameters) + bigRegTerm ==
        Approx(svmfBigReg.Evaluate(parameters)).epsilon(1e-7));
  }
}

/**
 * Test individual Evaluate() functions to be used for
 * optimization.
 */
TEST_CASE("LinearSVMFunctionSeparableEvaluate", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  LinearSVMFunction<> svmf(data, labels, numClasses);

  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    double hingeLoss = 0;
    for (size_t j = 0; j < points; ++j)
      hingeLoss += svmf.Evaluate(parameters, j, 1);

    hingeLoss /= points;

    // Compare with the value returned by the function.
    REQUIRE(svmf.Evaluate(parameters) == Approx(hingeLoss).epsilon(1e-7));
  }
}

/**
 *
 * Test regularization for the separable Evaluate() function
 * to be used Optimizers.
 */
TEST_CASE("LinearSVMFunctionRegularizationSeparableEvaluate", "[LinearSVMTest]")
{
  const size_t points = 100;
  const size_t trials = 3;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  LinearSVMFunction<> svmfNoReg(data, labels, numClasses, 0.0);
  LinearSVMFunction<> svmfSmallReg(data, labels, numClasses, 0.5);
  LinearSVMFunction<> svmfBigReg(data, labels, numClasses, 20.0);


  // Check that the number of functions is correct.
  REQUIRE(svmfNoReg.NumFunctions() == points);
  REQUIRE(svmfSmallReg.NumFunctions() == points);
  REQUIRE(svmfBigReg.NumFunctions() == points);


  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    double wL2SquaredNorm;
    wL2SquaredNorm = 0.5 * dot(parameters, parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 20 * wL2SquaredNorm;

    for (size_t j = 0; j < points; ++j)
    {
      REQUIRE(svmfNoReg.Evaluate(parameters, j, 1) + smallRegTerm ==
          Approx(svmfSmallReg.Evaluate(parameters, j, 1)).epsilon(1e-7));
      REQUIRE(svmfNoReg.Evaluate(parameters, j, 1) + bigRegTerm ==
          Approx(svmfBigReg.Evaluate(parameters, j, 1)).epsilon(1e-7));
    }
  }
}

/**
 * Test Gradient() of the LinearSVMFunction.
 */
TEST_CASE("LinearSVMFunctionGradient", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 5;
  const double delta = 1.0;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
                                    0.0 /* no regularization */,
                                    delta);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    // Hand-calculate the gradient.
    arma::mat difference;
    difference.zeros(numClasses, points);

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters.t() * data.col(j);
      double correct = score[labels(j)];
      size_t differenceCount = 0;
      for (size_t k = 0; k < numClasses; ++k)
      {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + delta;
        if (margin > 0)
        {
          differenceCount += 1;
          difference(k, j) = 1;
        }
      }
      difference(labels(j), j) -= differenceCount;
    }

    arma::mat gradient = (data * difference.t()) / points;
    arma::mat evaluatedGradient;

    svmf.Gradient(parameters, evaluatedGradient);

    // Compare with the values returned by Gradient().
    for (size_t j = 0; j < inputSize ; ++j)
    {
      for (size_t k = 0; k < numClasses ; ++k)
      {
        REQUIRE(gradient(j, k) ==
            Approx(evaluatedGradient(j, k)).epsilon(1e-7));
      }
    }
  }
}

/**
 * Test separable Gradient() of the LinearSVMFunction when regularization
 * is used.
 */
TEST_CASE("LinearSVMFunctionSeparableGradient", "[LinearSVMTest]")
{
  const size_t points = 100;
  const size_t trials = 3;
  const size_t inputSize = 5;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  LinearSVMFunction<> svmfNoReg(data, labels, numClasses, 0.0);
  LinearSVMFunction<> svmfSmallReg(data, labels, numClasses, 0.5);
  LinearSVMFunction<> svmfBigReg(data, labels, numClasses, 20.0);

  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(inputSize, numClasses);

    arma::mat gradient;
    arma::mat smallRegGradient;
    arma::mat bigRegGradient;

    // Test separable gradient for each point.  Regularization will be the same.
    for (size_t k = 0; k < points; ++k)
    {
      svmfNoReg.Gradient(parameters, k, gradient, 1);
      svmfSmallReg.Gradient(parameters, k, smallRegGradient, 1);
      svmfBigReg.Gradient(parameters, k, bigRegGradient, 1);

      // Check sizes of gradients.
      REQUIRE(gradient.n_elem == parameters.n_elem);
      REQUIRE(smallRegGradient.n_elem == parameters.n_elem);
      REQUIRE(bigRegGradient.n_elem == parameters.n_elem);

      // Check other terms.
      for (size_t j = 0; j < parameters.n_elem; ++j)
      {
        const double smallRegTerm = 0.5 * parameters[j];
        const double bigRegTerm = 20.0 * parameters[j];

        REQUIRE(gradient[j] + smallRegTerm ==
            Approx(smallRegGradient[j]).epsilon(1e-7));
        REQUIRE(gradient[j] + bigRegTerm ==
            Approx(bigRegGradient[j]).epsilon(1e-7));
      }
    }
  }
}

/**
 * Test training of linear svm on a simple dataset using
 * L-BFGS optimizer
 */
TEST_CASE("LinearSVMLBFGSSimpleTest", "[LinearSVMTest]")
{
  const size_t numClasses = 2;
  const double lambda = 0.0001;

  // A very simple fake dataset.
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  // Corresponding labels.
  arma::Row<size_t> labels = "1 0 1";

  // Create a linear svm object using L-BFGS optimizer.
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, lambda);

  // Compare training accuracy to 1.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  REQUIRE(acc == Approx(100.0).epsilon(0.005));
}

/**
 * Test training of linear svm on a simple dataset using
 * Gradient Descent optimizer
 */
TEST_CASE("LinearSVMGradientDescentSimpleTest", "[LinearSVMTest]")
{
  const size_t numClasses = 2;
  const size_t maxIterations = 10000;
  const double stepSize = 0.01;
  const double tolerance = 1e-5;
  const double lambda = 0.0001;
  const double delta = 1.0;

  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  // Create a linear svm object using custom gradient descent optimizer.
  ens::GradientDescent optimizer(stepSize, maxIterations, tolerance);
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, optimizer, lambda,
      delta, false);

  // Compare training accuracy to 1.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  REQUIRE(acc == Approx(100.0).epsilon(0.005));
}

/**
 * Test training of linear svm for two classes on a complex gaussian dataset
 * using L-BFGS optimizer.
 */
TEST_CASE("LinearSVMLBFGSTwoClasses", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;

  // Generate two-Gaussian dataset.
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0"),
      arma::eye<arma::mat>(3, 3));
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0"),
      arma::eye<arma::mat>(3, 3));

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  // This loop can be removed when ensmallen PR #136 is merged into a version
  // of ensmallen that is the minimum required ensmallen version for mlpack.
  // Basically, L-BFGS can sometimes have a condition that causes a failure in
  // optimization, and this is a workaround that runs multiple trials to avoid
  // that situation.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels(i) = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels(i) = 1;
    }

    // Create a linear svm object using L-BFGS optimizer.
    LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda);

    // Compare training accuracy to 1.
    const double acc = lsvm.ComputeAccuracy(data, labels);
    if (acc < 0.99)
    {
      continue; // This trial has failed.
    }

    // Create test dataset.
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels(i) = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels(i) = 1;
    }

    // Compare test accuracy to 1.
    const double testAcc = lsvm.ComputeAccuracy(data, labels);
    if (testAcc >= 0.99)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Test training of linear svm for two classes on a complex gaussian dataset
 * using L-BFGS optimizer which can't be separated without adding
 * the intercept term.
 */
TEST_CASE("LinearSVMFitIntercept", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;
  const double delta = 1.0;

  // Generate a two-Gaussian dataset,
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0"),
      arma::eye<arma::mat>(3, 3));
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0"),
      arma::eye<arma::mat>(3, 3));

  // This loop can be removed when ensmallen PR #136 is merged into a version
  // of ensmallen that is the minimum required ensmallen version for mlpack.
  // Basically, L-BFGS can sometimes have a condition that causes a failure in
  // optimization, and this is a workaround that runs multiple trials to avoid
  // that situation.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat data(inputSize, points);
    arma::Row<size_t> labels(points);
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels[i] = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels[i] = 1;
    }

    // Now train a svm object on it.
    LinearSVM<arma::mat> svm(data, labels, numClasses, lambda, delta, true);

    // Ensure that the error is close to zero.
    const double acc = svm.ComputeAccuracy(data, labels);
    if (acc <= 0.98)
      continue;

    arma::mat testData(inputSize, points);
    arma::Row<size_t> testLabels(inputSize);

    // Create a test set.
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels[i] = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels[i] = 1;
    }

    // Ensure that the error is close to zero.
    const double testAcc = svm.ComputeAccuracy(data, labels);
    if (testAcc >= 0.95)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Test training of linear svm on a simple dataset using
 * Gradient Descent optimizer and with another value of delta.
 */
TEST_CASE("LinearSVMDeltaLBFGSTwoClasses", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;
  const double delta = 5.0;

  // Generate two-Gaussian dataset.
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0"),
      arma::eye<arma::mat>(3, 3));
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0"),
      arma::eye<arma::mat>(3, 3));

  // This loop can be removed when ensmallen PR #136 is merged into a version
  // of ensmallen that is the minimum required ensmallen version for mlpack.
  // Basically, L-BFGS can sometimes have a condition that causes a failure in
  // optimization, and this is a workaround that runs multiple trials to avoid
  // that situation.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    arma::mat data(inputSize, points);
    arma::Row<size_t> labels(points);

    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels(i) = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels(i) = 1;
    }

    // Create a linear svm object using L-BFGS optimizer.
    LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda,
        delta);

    // Compare training accuracy to 1.
    const double acc = lsvm.ComputeAccuracy(data, labels);
    if (acc <= 0.99)
      continue;

    arma::mat testData(inputSize, points);
    arma::Row<size_t> testLabels(points);

    for (size_t i = 0; i < points / 2; ++i)
    {
      testData.col(i) = g1.Random();
      testLabels(i) = 0;
    }

    for (size_t i = points / 2; i < points; ++i)
    {
      testData.col(i) = g2.Random();
      testLabels(i) = 1;
    }

    // Compare test accuracy to 1.
    const double testAcc = lsvm.ComputeAccuracy(testData, testLabels);
    if (testAcc >= 0.95)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * The test is only compiled if the user has specified OpenMP to be
 * used.
 */
#ifdef MLPACK_USE_OPENMP

/**
 * Test training of linear svm on a simple dataset using
 * Parallel SGD optimizer.
 */
TEST_CASE("LinearSVMPSGDSimpleTest", "[LinearSVMTest]")
{
  const size_t numClasses = 2;
  const double lambda = 0.5;
  const double alpha = 0.01;
  const double delta = 1.0;

  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  ens::ConstantStep decayPolicy(alpha);

  // Train linear svm object using Parallel SGD optimizer.
  // The threadShareSize is chosen such that each function gets optimized.
  ens::ParallelSGD<ens::ConstantStep> optimizer(0,
      std::ceil((float) dataset.n_cols / omp_get_max_threads()),
      1e-5, true, decayPolicy);
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, optimizer, lambda,
      delta, false);

  // Compare training accuracy to 1.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  REQUIRE(acc == Approx(100.0).epsilon(1e-2));
}

/**
 * Test training of linear svm for two classes on a complex gaussian dataset
 * using Parallel SGD optimizer.
 */
TEST_CASE("LinearSVMParallelSGDTwoClasses", "[LinearSVMTest]")
{
  const size_t points = 500;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;
  const double alpha = 0.01;
  const double delta = 1.0;

  // Generate two-Gaussian dataset.
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0"),
      arma::eye<arma::mat>(3, 3));
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0"),
      arma::eye<arma::mat>(3, 3));

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 2; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 2; i < points; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // We run the test multiple times, since it sometimes fails, in order to get
  // the probability of failure down.
  bool success = false;
  const size_t trials = 4;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    ens::ConstantStep decayPolicy(alpha);

    // Train linear svm object using Parallel SGD optimizer.
    // The threadShareSize is chosen such that each function gets optimized.
    ens::ParallelSGD<ens::ConstantStep> optimizer(100000,
        std::ceil((float) data.n_cols / omp_get_max_threads()),
        1e-5, true, decayPolicy);
    LinearSVM<arma::mat> lsvm(data, labels, numClasses, optimizer, lambda,
        delta, false);

    // Compare training accuracy to 100.
    const double acc = lsvm.ComputeAccuracy(data, labels);

    // Create test dataset.
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels(i) = 0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels(i) = 1;
    }

    // Compare test accuracy to 1.
    const double testAcc = lsvm.ComputeAccuracy(data, labels);

    // Larger tolerance is sometimes needed.
    if (testAcc == Approx(100.0).epsilon(0.02) &&
        acc == Approx(100.0).epsilon(0.02))
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

#endif

/**
 * Test sparse and dense linear svm training and make sure they both work the
 * same using the L-BFGS optimizer.
 */
TEMPLATE_TEST_CASE("LinearSVMSparseLBFGSTest", "[LinearSVMTest]", float, double)
{
  using ElemType = TestType;
  using SparseMatType = arma::SpMat<ElemType>;
  using MatType = arma::Mat<ElemType>;

  // Create a random dataset.
  SparseMatType dataset;
  dataset.sprandu(10, 800, 0.3);
  MatType denseDataset(dataset);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LinearSVM<> lr(denseDataset, labels, 2, 0.3, 1, false);
  LinearSVM<> lrSparse(dataset, labels, 2, 0.3, 1, false);

  // Make the initial points the same.
  lrSparse.Parameters() = lr.Parameters();

  REQUIRE(lr.Parameters().n_elem == lrSparse.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
  {
    REQUIRE(lr.Parameters()[i] == Approx(lrSparse.Parameters()[i]).
        epsilon(5e-6));
  }
}

/**
 * Test training of linear svm for multiple classes on a complex gaussian
 * dataset using L-BFGS optimizer, with different types.
 */
TEMPLATE_TEST_CASE("LinearSVMLBFGSMultipleClasses", "[LinearSVMTest]", float,
    double)
{
  using ElemType = TestType;
  using MatType = arma::Mat<ElemType>;
  using VecType = arma::Col<ElemType>;

  const size_t points = 1000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  MatType identity = arma::eye<MatType>(5, 5);
  GaussianDistribution<MatType> g1(VecType("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<MatType> g2(VecType("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<MatType> g3(VecType("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<MatType> g4(VecType("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<MatType> g5(VecType("1.0 0.0 1.0 8.0 3.0"), identity);

  MatType data(inputSize, points);
  arma::Row<size_t> labels(points);

  // This loop can be removed when ensmallen PR #136 is merged into a version
  // of ensmallen that is the minimum required ensmallen version for mlpack.
  // Basically, L-BFGS can sometimes have a condition that causes a failure in
  // optimization, and this is a workaround that runs multiple trials to avoid
  // that situation.
  bool success = false;
  for (size_t trial = 0; trial < 5; ++trial)
  {
    for (size_t i = 0; i < points / 5; ++i)
    {
      data.col(i) = g1.Random();
      labels(i) = 0;
    }
    for (size_t i = points / 5; i < (2 * points) / 5; ++i)
    {
      data.col(i) = g2.Random();
      labels(i) = 1;
    }
    for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
    {
      data.col(i) = g3.Random();
      labels(i) = 2;
    }
    for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
    {
      data.col(i) = g4.Random();
      labels(i) = 3;
    }
    for (size_t i = (4 * points) / 5; i < points; ++i)
    {
      data.col(i) = g5.Random();
      labels(i) = 4;
    }

    // Train linear svm object using L-BFGS optimizer.
    LinearSVM<MatType> lsvm(data, labels, numClasses, lambda);

    // Compare training accuracy to 1.
    const double acc = lsvm.ComputeAccuracy(data, labels);
    if (acc <= 0.98)
      continue;

    // Create test dataset.
    for (size_t i = 0; i < points / 5; ++i)
    {
      data.col(i) = ConvTo<VecType>::From(g1.Random());
      labels(i) = 0;
    }
    for (size_t i = points / 5; i < (2 * points) / 5; ++i)
    {
      data.col(i) = ConvTo<VecType>::From(g2.Random());
      labels(i) = 1;
    }
    for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
    {
      data.col(i) = ConvTo<VecType>::From(g3.Random());
      labels(i) = 2;
    }
    for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
    {
      data.col(i) = ConvTo<VecType>::From(g4.Random());
      labels(i) = 3;
    }
    for (size_t i = (4 * points) / 5; i < points; ++i)
    {
      data.col(i) = ConvTo<VecType>::From(g5.Random());
      labels(i) = 4;
    }

    // Compare test accuracy to 1.
    const double testAcc = lsvm.ComputeAccuracy(data, labels);
    if (testAcc >= 0.98)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

/**
 * Testing single point classification (Classify()).
 */
TEMPLATE_TEST_CASE("LinearSVMClassifySinglePointTest", "[LinearSVMTest]", float,
    double)
{
  using ElemType = TestType;
  using MatType = arma::Mat<ElemType>;
  using VecType = arma::Col<ElemType>;

  const size_t points = 500;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  MatType identity = arma::eye<MatType>(5, 5);
  GaussianDistribution<MatType> g1(VecType("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<MatType> g2(VecType("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<MatType> g3(VecType("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<MatType> g4(VecType("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<MatType> g5(VecType("1.0 0.0 1.0 8.0 3.0"), identity);

  MatType data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train linear svm object.
  LinearSVM<MatType> lsvm(data, labels, numClasses, lambda);

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  MatType scores;
  lsvm.Classify(data, labels, scores);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(lsvm.Classify(data.col(i)) == labels(i));

    size_t prediction;
    VecType scoresVec;
    lsvm.Classify(data.col(i), prediction, scoresVec);

    REQUIRE(prediction == labels(i));
    REQUIRE(scoresVec.n_elem == scores.n_rows);
    REQUIRE(arma::approx_equal(scoresVec, scores.col(i), "absdiff", 1e-5));
  }
}

/**
 * Test that single-point classification gives the same results as multi-point
 * classification.
 */
TEST_CASE("SinglePointClassifyTest", "[LinearSVMTest]")
{
  const size_t points = 500;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<> g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<> g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<> g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train linear svm object.
  LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda);

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  arma::Row<size_t> predictions;
  lsvm.Classify(data, predictions);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t pred = lsvm.Classify(data.col(i));

    REQUIRE(pred == predictions[i]);
  }
}

/**
 * Test a Linear SVM model with EndOptimization callback.
 */
TEST_CASE("LinearSVMCallbackTest", "[LinearSVMTest]")
{
  const size_t numClasses = 2;
  const double lambda = 0.0001;
  const double delta = 1.0;

  // A very simple fake dataset.
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  // Corresponding labels.
  arma::Row<size_t> labels = "1 0 1";

  CallbackTestFunction cb;

  ens::L_BFGS opt;
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, lambda, delta,
      false, cb);

  REQUIRE(cb.calledEndOptimization == true);
}

// Test all variants of LinearSVM constructors.
TEMPLATE_TEST_CASE("LinearSVMConstructorVariantTest", "[LinearSVMTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;

  // Create some random data.  The results here do not matter all that much;
  // this is more of a test that all constructor variants successfully compile
  // and produce models at all.
  MatType dataset(10, 800, arma::fill::randu);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LinearSVM<> lsvm1;
  LinearSVM<> lsvm2(10, 2);
  LinearSVM<> lsvm3(10, 2, 0.0002, 1.1, true);
  LinearSVM<> lsvm4(dataset, labels, 2);
  LinearSVM<> lsvm5(dataset, labels, 2, 0.0003, 1.2, true);
  LinearSVM<> lsvm6(dataset, labels, 2, 0.0004, 1.3, true,
      CallbackTestFunction());
  LinearSVM<> lsvm7(dataset, labels, 2, 0.0005, 1.4, true,
      CallbackTestFunction(), ens::TimerStop(1.0));

  ens::StandardSGD sgd;
  LinearSVM<> lsvm8(dataset, labels, 2, sgd);
  LinearSVM<> lsvm9(dataset, labels, 2, sgd, 0.0006, 1.5, true);
  LinearSVM<> lsvm10(dataset, labels, 2, sgd, 0.0007, 1.6, true,
      CallbackTestFunction());
  LinearSVM<> lsvm11(dataset, labels, 2, sgd, 0.0008, 1.7, true,
      CallbackTestFunction(), ens::TimerStop(1.0));

  // Check that the variants that did not train have reasonable values.
  REQUIRE(lsvm1.Lambda() == Approx(0.0001));
  REQUIRE(lsvm1.Delta() == Approx(1.0));
  REQUIRE(lsvm1.FitIntercept() == false);

  REQUIRE(lsvm2.Lambda() == Approx(0.0001));
  REQUIRE(lsvm2.Delta() == Approx(1.0));
  REQUIRE(lsvm2.FitIntercept() == false);
  REQUIRE(lsvm2.NumClasses() == 2);

  REQUIRE(lsvm3.Lambda() == Approx(0.0002));
  REQUIRE(lsvm3.Delta() == Approx(1.1));
  REQUIRE(lsvm3.FitIntercept() == true);
  REQUIRE(lsvm3.NumClasses() == 2);

  // Now check that the variants that trained have reasonable models of the
  // right size.
  REQUIRE(lsvm4.Lambda() == Approx(0.0001));
  REQUIRE(lsvm4.Delta() == Approx(1.0));
  REQUIRE(lsvm4.FitIntercept() == false);
  REQUIRE(lsvm4.FeatureSize() == 10);
  REQUIRE(lsvm4.NumClasses() == 2);

  REQUIRE(lsvm5.Lambda() == Approx(0.0003));
  REQUIRE(lsvm5.Delta() == Approx(1.2));
  REQUIRE(lsvm5.FitIntercept() == true);
  REQUIRE(lsvm5.FeatureSize() == 10);
  REQUIRE(lsvm5.NumClasses() == 2);

  REQUIRE(lsvm6.Lambda() == Approx(0.0004));
  REQUIRE(lsvm6.Delta() == Approx(1.3));
  REQUIRE(lsvm6.FitIntercept() == true);
  REQUIRE(lsvm6.FeatureSize() == 10);
  REQUIRE(lsvm6.NumClasses() == 2);

  REQUIRE(lsvm7.Lambda() == Approx(0.0005));
  REQUIRE(lsvm7.Delta() == Approx(1.4));
  REQUIRE(lsvm7.FitIntercept() == true);
  REQUIRE(lsvm7.FeatureSize() == 10);
  REQUIRE(lsvm7.NumClasses() == 2);

  REQUIRE(lsvm8.Lambda() == Approx(0.0001));
  REQUIRE(lsvm8.Delta() == Approx(1.0));
  REQUIRE(lsvm8.FitIntercept() == false);
  REQUIRE(lsvm8.FeatureSize() == 10);
  REQUIRE(lsvm8.NumClasses() == 2);

  REQUIRE(lsvm9.Lambda() == Approx(0.0006));
  REQUIRE(lsvm9.Delta() == Approx(1.5));
  REQUIRE(lsvm9.FitIntercept() == true);
  REQUIRE(lsvm9.FeatureSize() == 10);
  REQUIRE(lsvm9.NumClasses() == 2);

  REQUIRE(lsvm10.Lambda() == Approx(0.0007));
  REQUIRE(lsvm10.Delta() == Approx(1.6));
  REQUIRE(lsvm10.FitIntercept() == true);
  REQUIRE(lsvm10.FeatureSize() == 10);
  REQUIRE(lsvm10.NumClasses() == 2);

  REQUIRE(lsvm11.Lambda() == Approx(0.0008));
  REQUIRE(lsvm11.Delta() == Approx(1.7));
  REQUIRE(lsvm11.FitIntercept() == true);
  REQUIRE(lsvm11.FeatureSize() == 10);
  REQUIRE(lsvm11.NumClasses() == 2);
}

// Test all variants of LinearSVM Train() functions.
TEMPLATE_TEST_CASE("LinearSVMTrainVariantTest", "[LinearSVMTest]", arma::fmat,
    arma::mat)
{
  using MatType = TestType;

  // Create some random data.  The results here do not matter all that much;
  // this is more of a test that all constructor variants successfully compile
  // and produce models at all.
  MatType dataset(10, 800, arma::fill::randu);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LinearSVM<> lsvm1(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm2(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm3(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm4(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm5(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm6(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm7(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm8(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm9(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm10(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm11(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm12(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm13(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm14(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm15(10, 2, 1.2, 0.5, true);
  LinearSVM<> lsvm16(10, 2, 1.2, 0.5, true);

  lsvm1.Train(dataset, labels, 2);
  lsvm2.Train(dataset, labels, 2, CallbackTestFunction());
  lsvm3.Train(dataset, labels, 2, CallbackTestFunction(), ens::TimerStop(1.0));
  lsvm4.Train(dataset, labels, 2, 0.0002);
  lsvm5.Train(dataset, labels, 2, 0.0003, 1.1);
  lsvm6.Train(dataset, labels, 2, 0.0004, 1.2, false);
  lsvm7.Train(dataset, labels, 2, 0.0005, 1.3, true, CallbackTestFunction());
  lsvm8.Train(dataset, labels, 2, 0.0006, 1.4, false, CallbackTestFunction(),
      ens::TimerStop(1.0));

  ens::Adam adam;
  lsvm9.Train(dataset, labels, 2, adam);
  lsvm10.Train(dataset, labels, 2, adam, CallbackTestFunction());
  lsvm11.Train(dataset, labels, 2, adam, CallbackTestFunction(),
      ens::TimerStop(1.0));
  lsvm12.Train(dataset, labels, 2, adam, 0.0007);
  lsvm13.Train(dataset, labels, 2, adam, 0.0008, 1.5);
  lsvm14.Train(dataset, labels, 2, adam, 0.0009, 1.6, false);
  lsvm15.Train(dataset, labels, 2, adam, 0.001, 1.7, true,
      CallbackTestFunction());
  lsvm16.Train(dataset, labels, 2, adam, 0.0011, 1.8, false,
      CallbackTestFunction(), ens::TimerStop(1.0));

  // Check that all hyperparameters are set as expected, and that the model has
  // the correct size.
  REQUIRE(lsvm1.Lambda() == Approx(1.2));
  REQUIRE(lsvm1.Delta() == Approx(0.5));
  REQUIRE(lsvm1.FitIntercept() == true);
  REQUIRE(lsvm1.FeatureSize() == 10);
  REQUIRE(lsvm1.NumClasses() == 2);

  REQUIRE(lsvm2.Lambda() == Approx(1.2));
  REQUIRE(lsvm2.Delta() == Approx(0.5));
  REQUIRE(lsvm2.FitIntercept() == true);
  REQUIRE(lsvm2.FeatureSize() == 10);
  REQUIRE(lsvm2.NumClasses() == 2);

  REQUIRE(lsvm3.Lambda() == Approx(1.2));
  REQUIRE(lsvm3.Delta() == Approx(0.5));
  REQUIRE(lsvm3.FitIntercept() == true);
  REQUIRE(lsvm3.FeatureSize() == 10);
  REQUIRE(lsvm3.NumClasses() == 2);

  REQUIRE(lsvm4.Lambda() == Approx(0.0002));
  REQUIRE(lsvm4.Delta() == Approx(0.5));
  REQUIRE(lsvm4.FitIntercept() == true);
  REQUIRE(lsvm4.FeatureSize() == 10);
  REQUIRE(lsvm4.NumClasses() == 2);

  REQUIRE(lsvm5.Lambda() == Approx(0.0003));
  REQUIRE(lsvm5.Delta() == Approx(1.1));
  REQUIRE(lsvm5.FitIntercept() == true);
  REQUIRE(lsvm5.FeatureSize() == 10);
  REQUIRE(lsvm5.NumClasses() == 2);

  REQUIRE(lsvm6.Lambda() == Approx(0.0004));
  REQUIRE(lsvm6.Delta() == Approx(1.2));
  REQUIRE(lsvm6.FitIntercept() == false);
  REQUIRE(lsvm6.FeatureSize() == 10);
  REQUIRE(lsvm6.NumClasses() == 2);

  REQUIRE(lsvm7.Lambda() == Approx(0.0005));
  REQUIRE(lsvm7.Delta() == Approx(1.3));
  REQUIRE(lsvm7.FitIntercept() == true);
  REQUIRE(lsvm7.FeatureSize() == 10);
  REQUIRE(lsvm7.NumClasses() == 2);

  REQUIRE(lsvm8.Lambda() == Approx(0.0006));
  REQUIRE(lsvm8.Delta() == Approx(1.4));
  REQUIRE(lsvm8.FitIntercept() == false);
  REQUIRE(lsvm8.FeatureSize() == 10);
  REQUIRE(lsvm8.NumClasses() == 2);

  REQUIRE(lsvm9.Lambda() == Approx(1.2));
  REQUIRE(lsvm9.Delta() == Approx(0.5));
  REQUIRE(lsvm9.FitIntercept() == true);
  REQUIRE(lsvm9.FeatureSize() == 10);
  REQUIRE(lsvm9.NumClasses() == 2);

  REQUIRE(lsvm10.Lambda() == Approx(1.2));
  REQUIRE(lsvm10.Delta() == Approx(0.5));
  REQUIRE(lsvm10.FitIntercept() == true);
  REQUIRE(lsvm10.FeatureSize() == 10);
  REQUIRE(lsvm10.NumClasses() == 2);

  REQUIRE(lsvm11.Lambda() == Approx(1.2));
  REQUIRE(lsvm11.Delta() == Approx(0.5));
  REQUIRE(lsvm11.FitIntercept() == true);
  REQUIRE(lsvm11.FeatureSize() == 10);
  REQUIRE(lsvm11.NumClasses() == 2);

  REQUIRE(lsvm12.Lambda() == Approx(0.0007));
  REQUIRE(lsvm12.Delta() == Approx(0.5));
  REQUIRE(lsvm12.FitIntercept() == true);
  REQUIRE(lsvm12.FeatureSize() == 10);
  REQUIRE(lsvm12.NumClasses() == 2);

  REQUIRE(lsvm13.Lambda() == Approx(0.0008));
  REQUIRE(lsvm13.Delta() == Approx(1.5));
  REQUIRE(lsvm13.FitIntercept() == true);
  REQUIRE(lsvm13.FeatureSize() == 10);
  REQUIRE(lsvm13.NumClasses() == 2);

  REQUIRE(lsvm14.Lambda() == Approx(0.0009));
  REQUIRE(lsvm14.Delta() == Approx(1.6));
  REQUIRE(lsvm14.FitIntercept() == false);
  REQUIRE(lsvm14.FeatureSize() == 10);
  REQUIRE(lsvm14.NumClasses() == 2);

  REQUIRE(lsvm15.Lambda() == Approx(0.001));
  REQUIRE(lsvm15.Delta() == Approx(1.7));
  REQUIRE(lsvm15.FitIntercept() == true);
  REQUIRE(lsvm15.FeatureSize() == 10);
  REQUIRE(lsvm15.NumClasses() == 2);

  REQUIRE(lsvm16.Lambda() == Approx(0.0011));
  REQUIRE(lsvm16.Delta() == Approx(1.8));
  REQUIRE(lsvm16.FitIntercept() == false);
  REQUIRE(lsvm16.FeatureSize() == 10);
  REQUIRE(lsvm16.NumClasses() == 2);
}
