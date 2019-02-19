/**
 * @file linear_svm_test.cpp
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
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(LinearSVMTest);

/**
 * A simple test for LinearSVMFunction
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionEvaluate)
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
                         " 1 1 1 1 1";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 1.0, 1e-5);

  parameters = "2 0 1 2 2;"
               "1 2 2 2 2";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 2.0, 1e-5);

  parameters = "-0.1425 8.3228 0.1724 -0.3374 0.1548;"
               "0.1435 0.0009 -0.1736 0.3356 -0.1544";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 0.0, 1e-5);

  parameters = "100 3 4 5 23;"
               "43 54 67 32 64";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 85.33333333, 1e-5);

  parameters = "3 71 22 12 6;"
               "100 39 30 57 22";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 11.0, 1e-5);
}

/**
 * A complicated test for the LinearSVMFunction for binary-class
 * classification.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRandomBinaryEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 2;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k)
      {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + 1;
        if (margin > 0)
          hingeLoss += margin;
      }
    }
    hingeLoss /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), hingeLoss, 1e-5);
  }
}

/**
 * A complicated test for the LinearSVMFunction for multi-class
 * classification.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRandomEvaluate)
{
  const size_t points = 500;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k)
      {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + 1;
        if (margin > 0)
          hingeLoss += margin;
      }
    }
    hingeLoss /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), hingeLoss, 1e-5);
  }
}

/**
 * Test regularization for the LinearSVMFunction Evaluate()
 * function.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRegularizationEvaluate)
{
  const size_t points = 500;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 3 objects for comparing regularization costs.
  LinearSVMFunction<arma::mat> svmfNoReg(data, labels, numClasses, 0);
  LinearSVMFunction<arma::mat> svmfSmallReg(data, labels, numClasses, 1);
  LinearSVMFunction<arma::mat> svmfBigReg(data, labels, numClasses, 20);

  // Run a number of trials.
  for (size_t i = 0; i < trials; i++)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double wL2SquaredNorm;
    wL2SquaredNorm = arma::accu(parameters % parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 10 * wL2SquaredNorm;

    BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters) + smallRegTerm,
                        svmfSmallReg.Evaluate(parameters), 1e-5);
    BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters) + bigRegTerm,
                        svmfBigReg.Evaluate(parameters), 1e-5);
  }
}

/**
 * Test individual Evaluate() functions to be used for
 * optimization.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionSeparableEvaluate)
{
  const size_t points = 500;
  const size_t trials = 10;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  LinearSVMFunction<> svmf(data, labels, numClasses);

  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double hingeLoss = 0;
    for (size_t j = 0; j < points; ++j)
      hingeLoss += svmf.Evaluate(parameters, j, 1);

    hingeLoss /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), hingeLoss, 1e-5);
  }
}

/**
 *
 * Test regularization for the separable Evaluate() function
 * to be used Optimizers.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRegularizationSeparableEvaluate)
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
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  LinearSVMFunction<> svmfNoReg(data, labels, numClasses, 0.0);
  LinearSVMFunction<> svmfSmallReg(data, labels, numClasses, 0.5);
  LinearSVMFunction<> svmfBigReg(data, labels, numClasses, 20.0);


  // Check that the number of functions is correct.
  BOOST_REQUIRE_EQUAL(svmfNoReg.NumFunctions(), points);
  BOOST_REQUIRE_EQUAL(svmfSmallReg.NumFunctions(), points);
  BOOST_REQUIRE_EQUAL(svmfBigReg.NumFunctions(), points);


  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double wL2SquaredNorm;
    wL2SquaredNorm = 0.5 * arma::accu(parameters % parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 20 * wL2SquaredNorm;

    for (size_t j = 0; j < points; ++j)
    {
      BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters, j, 1) + smallRegTerm,
          svmfSmallReg.Evaluate(parameters, j, 1), 1e-5);
      BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters, j, 1) + bigRegTerm,
          svmfBigReg.Evaluate(parameters, j, 1), 1e-5);
    }
  }
}

/**
 * Test gradient for the LinearSVMFunction.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionGradient)
{
  const size_t points = 500;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 2 objects for 2 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  LinearSVMFunction<arma::mat> svmf1(data, labels, numClasses, 0.0);
  LinearSVMFunction<arma::mat> svmf2(data, labels, numClasses, 10.0);

  // Create a random set of parameters.
  arma::mat parameters;
  parameters.randu(numClasses, inputSize);

  // Get gradients for the current parameters.
  arma::mat gradient1, gradient2;
  svmf1.Gradient(parameters, gradient1);
  svmf2.Gradient(parameters, gradient2);

  // Perturbation constant.
  const double epsilon = 0.001;
  double costPlus1, costMinus1, numGradient1;
  double costPlus2, costMinus2, numGradient2;


  // For each parameter.
  for (size_t i = 0; i < numClasses; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
    {
      // Perturb parameter with a positive constant and get costs.
      parameters(i, j) += epsilon;
      costPlus1 = svmf1.Evaluate(parameters);
      costPlus2 = svmf2.Evaluate(parameters);

      // Perturb parameter with a negative constant and get costs.
      parameters(i, j) -= 2 * epsilon;
      costMinus1 = svmf1.Evaluate(parameters);
      costMinus2 = svmf2.Evaluate(parameters);

      // Compute numerical gradients using the costs calculated above.
      numGradient1 = (costPlus1 - costMinus1) / (2 * epsilon);
      numGradient2 = (costPlus2 - costMinus2) / (2 * epsilon);

      // Restore the parameter value.
      parameters(i, j) += epsilon;

      // Compare numerical and backpropagation gradient values.
      BOOST_REQUIRE_SMALL(numGradient1 - gradient1(i, j), 1e-2);
      BOOST_REQUIRE_SMALL(numGradient2 - gradient2(i, j), 1e-2);
    }
  }
}

/**
 * Test separable Gradient() of the LinearSVMFunction when regularization
 * is used.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionSeparableGradient)
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
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  LinearSVMFunction<> svmfNoReg(data, labels, numClasses, 0.0);
  LinearSVMFunction<> svmfSmallReg(data, labels, numClasses, 0.5);
  LinearSVMFunction<> svmfBigReg(data, labels, numClasses, 20.0);

  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

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
      BOOST_REQUIRE_EQUAL(gradient.n_elem, parameters.n_elem);
      BOOST_REQUIRE_EQUAL(smallRegGradient.n_elem, parameters.n_elem);
      BOOST_REQUIRE_EQUAL(bigRegGradient.n_elem, parameters.n_elem);

      // Check other terms.
      for (size_t j = 0; j < parameters.n_elem; ++j)
      {
        const double smallRegTerm = 0.5 * parameters[j];
        const double bigRegTerm = 20.0 * parameters[j];

        BOOST_REQUIRE_CLOSE(gradient[j] + smallRegTerm, smallRegGradient[j],
                            1e-5);
        BOOST_REQUIRE_CLOSE(gradient[j] + bigRegTerm, bigRegGradient[j], 1e-5);
      }
    }
  }
}

/**
 * Test training of linear svm on a simple dataset using
 * Parallel SGD optimizer.
 */
BOOST_AUTO_TEST_CASE(LinearSVMPSGDSimpleTest)
{
  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  // Create a linear svm object using a custom Parallel
  // SGD object.
  ens::ParallelSGD<> psgd(1000, 3, 1e-5);
  LinearSVM<arma::mat> lsvm(dataset, labels, 2, 0.0001, psgd);

  // Compare training accuracy to 100.
  const double acc = lsvm.ComputeAccuracy(dataset, labels);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.5);
}

/**
 * Test training of linear svm on a simple dataset using
 * L-BFGS optimizer
 */
BOOST_AUTO_TEST_CASE(LinearSVMLGFGSSimpleTest)
{
    // A very simple fake dataset
    arma::mat dataset = "2 0 0;"
                        "0 0 0;"
                        "0 2 1;"
                        "1 0 2;"
                        "0 1 0";

    //  Corresponding labels
    arma::Row<size_t> labels = "1 0 1";

    // Create a linear svm object using a custom L-BFGS object.
    LinearSVM<arma::mat> lsvm(dataset, labels, 2, 0.0001, ens::L_BFGS());

    // Compare training accuracy to 100.
    const double acc = lsvm.ComputeAccuracy(dataset, labels);
    BOOST_REQUIRE_CLOSE(acc, 100.0, 0.5);
}

/**
 * Test training of linear svm for two classes on a complex gaussian dataset
 * using L-BFGS optimizer.
 */
BOOST_AUTO_TEST_CASE(LinearSVMLBFGSTwoClasses)
{
  const size_t points = 500;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;

  // Generate two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 2; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 2; i < points; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // Train linear svm object using L-BFGS optimizer.
  LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda, ens::L_BFGS());

  // Compare training accuracy to 100.
  const double acc = lsvm.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.5);

  // Create test dataset.
  for (size_t i = 0; i < points / 2; i++)
  {
    data.col(i) = g1.Random();
    labels(i) =  0;
  }
  for (size_t i = points / 2; i < points; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // Compare test accuracy to 100.
  const double testAcc = lsvm.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6);
}

/**
 * Test sparse and dense linear svm and make sure they both work the
 * same using the L-BFGS optimizer.
 */
BOOST_AUTO_TEST_CASE(LinearSVMSparseLBFGSTest)
{
  // Create a random dataset.
  arma::sp_mat dataset;
  dataset.sprandu(10, 800, 0.3);
  arma::mat denseDataset(dataset);
  arma::Row<size_t> labels(800);
  for (size_t i = 0; i < 800; ++i)
    labels[i] = math::RandInt(0, 2);

  LinearSVM<arma::mat> lr(denseDataset, labels, 2, 0.3, ens::L_BFGS());
  LinearSVM<arma::sp_mat> lrSparse(dataset, labels, 2, 0.3, ens::L_BFGS());

  BOOST_REQUIRE_EQUAL(lr.Parameters().n_elem, lrSparse.Parameters().n_elem);
  for (size_t i = 0; i < lr.Parameters().n_elem; ++i)
    BOOST_REQUIRE_CLOSE(lr.Parameters()[i], lrSparse.Parameters()[i], 5e-4);
}

/**
 * Test training of linear svm for multiple classes on a complex gaussian
 * dataset using L-BFGS optimizer.
 */
BOOST_AUTO_TEST_CASE(LinearSVMLBFGSMultipleClasses)
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

  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train linear svm object using L-BFGS optimizer.
  LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda, ens::L_BFGS());

  // Compare training accuracy to 100.
  const double acc = lsvm.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 2.0);

  // Create test dataset.
  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Compare test accuracy to 100.
  const double testAcc = lsvm.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 2.0);
}

/**
 * Testing Train() in LinearSVM.
 */
BOOST_AUTO_TEST_CASE(LinearSVMTrainTest)
{
  // Test the stability of the LinearSVM.
  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 500; ++i)
    labels[i] = size_t(0.0);
  for (size_t i = 500; i < 1000; ++i)
    labels[i] = size_t(1.0);

  LinearSVM<arma::mat> lsvm(dataset.n_rows, 2);
  LinearSVM<arma::mat> lsvm2(dataset.n_rows, 2);
  lsvm.Parameters() = lsvm2.Parameters();
  ens::L_BFGS lbfgs;
  lsvm.Train(dataset, labels, 2, std::move(lbfgs));
  lsvm2.Train(dataset, labels, 2, std::move(lbfgs));

  // Ensure that the parameters are the same.
  BOOST_REQUIRE_EQUAL(lsvm.Parameters().n_rows, lsvm2.Parameters().n_rows);
  BOOST_REQUIRE_EQUAL(lsvm.Parameters().n_cols, lsvm2.Parameters().n_cols);
  for (size_t i = 0; i < lsvm.Parameters().n_elem; ++i)
  {
    if (std::abs(lsvm.Parameters()[i]) < 1e-4)
      BOOST_REQUIRE_SMALL(lsvm2.Parameters()[i], 1e-4);
    else
      BOOST_REQUIRE_CLOSE(lsvm.Parameters()[i], lsvm2.Parameters()[i], 1e-4);
  }
}

/**
 * Testing single point classification (Classify()).
 */
BOOST_AUTO_TEST_CASE(LinearSVMClassifySinglePointTest)
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

  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train linear svm object.
  LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda, ens::L_BFGS());

  // Create test dataset.
  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  lsvm.Classify(data, labels);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    BOOST_REQUIRE_EQUAL(lsvm.Classify(data.col(i)), labels(i));
  }
}

/**
 * Test multi-point classification (Classify()).
 */
BOOST_AUTO_TEST_CASE(LinearSVMClassifyTest)
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

  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train linear svm object.
  LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda, ens::L_BFGS());

  // Create test dataset.
  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  arma::Row<size_t> predictions;
  lsvm.Classify(data, predictions);

  BOOST_REQUIRE_GE((double) arma::accu(predictions == labels), 450);
}

/**
 * Test that single-point classification gives the same results as multi-point
 * classification.
 */
BOOST_AUTO_TEST_CASE(SinglePointClassifyTest)
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

  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train linear svm object.
  LinearSVM<arma::mat> lsvm(data, labels, numClasses, lambda, ens::L_BFGS());

  // Create test dataset.
  for (size_t i = 0; i < points / 5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  arma::Row<size_t> predictions;
  lsvm.Classify(data, predictions);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    size_t pred = lsvm.Classify(data.col(i));

    BOOST_REQUIRE_EQUAL(pred, predictions[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
