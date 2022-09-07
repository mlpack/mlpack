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
    wL2SquaredNorm = arma::dot(parameters, parameters);

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
    wL2SquaredNorm = 0.5 * arma::dot(parameters, parameters);

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
  REQUIRE(acc == Approx(1.0).epsilon(0.005));
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
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, lambda,
      delta, false, optimizer);

  // Compare training accuracy to 1.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  REQUIRE(acc == Approx(1.0).epsilon(0.005));
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
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

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
      labels(i) =  0;
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
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

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
    LinearSVM<arma::mat> svm(data, labels, numClasses, lambda,
        delta, true, ens::L_BFGS());

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
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

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
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, lambda,
      delta, false, optimizer);

  // Compare training accuracy to 1.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  REQUIRE(acc == Approx(1.0).epsilon(1e-2));
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
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

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
    ens::ParallelSGD<ens::ConstantStep> optimizer(0,
        std::ceil((float) data.n_cols / omp_get_max_threads()),
        1e-5, true, decayPolicy);
    LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda,
        delta, false, optimizer);

    // Compare training accuracy to 1.
    const double acc = lsvm.ComputeAccuracy(data, labels);

    // Create test dataset.
    for (size_t i = 0; i < points / 2; ++i)
    {
      data.col(i) = g1.Random();
      labels(i) =  0;
    }
    for (size_t i = points / 2; i < points; ++i)
    {
      data.col(i) = g2.Random();
      labels(i) = 1;
    }

    // Compare test accuracy to 1.
    const double testAcc = lsvm.ComputeAccuracy(data, labels);

    // Larger tolerance is sometimes needed.
    if (testAcc == Approx(1.0).epsilon(0.02) &&
        acc == Approx(1.0).epsilon(0.02))
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

#endif

/**
 * Test sparse and dense linear svm and make sure they both work the
 * same using the L-BFGS optimizer.
 */
TEST_CASE("LinearSVMSparseLBFGSTest", "[LinearSVMTest]")
{
  // Create a random dataset.
  arma::sp_mat dataset;
  dataset.sprandu(10, 800, 0.3);
  arma::mat denseDataset(dataset);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = RandInt(0, 2);

  LinearSVM<arma::mat> lr(denseDataset, labels, 2, 0.3, 1,
      false, ens::L_BFGS());
  LinearSVM<arma::sp_mat> lrSparse(dataset, labels, 2, 0.3, 1,
      false, ens::L_BFGS());

  REQUIRE(lr.Parameters().n_elem == lrSparse.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
  {
    REQUIRE(lr.Parameters()[i] == Approx(lrSparse.Parameters()[i]).
        epsilon(5e-6));
  }
}

/**
 * Test training of linear svm for multiple classes on a complex gaussian
 * dataset using L-BFGS optimizer.
 */
TEST_CASE("LinearSVMLBFGSMultipleClasses", "[LinearSVMTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

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
    LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda);

    // Compare training accuracy to 1.
    const double acc = lsvm.ComputeAccuracy(data, labels);
    if (acc <= 0.98)
      continue;

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
TEST_CASE("LinearSVMClassifySinglePointTest", "[LinearSVMTest]")
{
  const size_t points = 500;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

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

  lsvm.Classify(data, labels);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(lsvm.Classify(data.col(i)) == labels(i));
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
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

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
  LinearSVM<arma::mat> lsvm(dataset, labels, numClasses, lambda,
      delta, false, opt, cb);

  REQUIRE(cb.calledEndOptimization == true);
}
