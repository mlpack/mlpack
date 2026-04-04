/**
 * @file tests/main_tests/nca_test.cpp
 * @author Yasmine Dumouchel
 *
 * Test RUN_BINDING() of nca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/methods/nca/nca_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(NCATestFixture);

/**
 * Ensure that, when labels are implicitily given with input,
 * the last column is treated as labels and that we get the
 * desired shape of output.
 */
TEST_CASE_METHOD(NCATestFixture, "NCAExplicitImplicitLabelsTest",
                "[NCAMainTest][BindingTests]")
{
  // Random dataset.
  arma::mat x;
  x.randu(3, 6);

  SetInputParam("input", std::move(x));
  SetInputParam("max_iterations", 100);

  RUN_BINDING();

  // Check that last row was treated as label by checking that
  // the output has 1 less row.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 2);
  REQUIRE(params.Get<arma::mat>("output").n_cols == 2);

  // Reset Settings.
  CleanMemory();
  ResetSettings();

  // Now check that when labels are explicitely given, the last column
  // of input is not treated as labels.
  arma::mat y              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels));
  SetInputParam("max_iterations", 100);

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 2);
  REQUIRE(params.Get<arma::mat>("output").n_cols == 2);
}

/**
 * Ensure that when we pass optimizer of type lbfgs, we also get the desired
 * shape of output.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALBFGSTest",
                "[NCAMainTest][BindingTests]")
{
  arma::mat x;
  x.randu(3, 100);
  arma::Row<size_t> labels;
  labels.zeros(100);

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));

  RUN_BINDING();

  // Check that final output has expected number of rows and colums.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 3);
  REQUIRE(params.Get<arma::mat>("output").n_cols == 3);
}

/**
 * Ensure that if labels are of a different size than required
 * by the input, an error occurs.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALabelSizeTest",
                "[NCAMainTest][BindingTests]")
{
  // Input labels of wrong size.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    1    2 ";

  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));

  // Check that an error is thrown.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that setting normalize as true results in a
 * different output matrix then when set to false.
 */
TEST_CASE_METHOD(NCATestFixture, "NCANormalizationTest",
                "[NCAMainTest][BindingTests]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  arma::mat inputData;
  if (!Load("vc2.csv", inputData))
    FAIL("Cannot load vc2.csv!");

  arma::Row<size_t> labels;
  if (!Load("vc2_labels.txt", labels))
    FAIL("Cannot load vc2_labels.txt!");

  // Set parameters and set normalize to true.
  SetInputParam("input", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset rettings.
  CleanMemory();
  ResetSettings();

  arma::mat inputData2;
  if (!Load("vc2.csv", inputData2))
    FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels2;
  if (!Load("vc2_labels.txt", labels2))
    FAIL("Cannot load vc2_labels.txt!");

  // Use the same input but set normalize to false.
  SetInputParam("input", std::move(inputData2));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("normalize", true);
  SetInputParam("linear_scan", true);
  SetInputParam("tolerance", 0.01);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that output is different when step_size is different.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentStepSizeTest",
                "[NCAMainTest][BindingTests]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Simple dataset with 6 points and two classes.
  arma::mat x              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Set parameters with a small step_size.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("step_size", (double) 1.2);
  SetInputParam("linear_scan", true);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset settings.
  CleanMemory();
  ResetSettings();

  // Same dataset.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but with a larger step_size.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("step_size", (double) 20.5);
  SetInputParam("linear_scan", true);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that output is different when the tolerance is different.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentToleranceTest",
                "[NCAMainTest][BindingTests][long]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Random dataset.
  arma::mat x;
  x.randu(3, 600);
  arma::Row<size_t> labels = arma::randi<arma::Row<size_t>>(600,
      DistrParam(0, 1));

  arma::mat y = x;
  arma::Row<size_t> labels2 = labels;

  // Set parameters with a small tolerance.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer", std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 0);
  SetInputParam("tolerance", (double) 0.1);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset settings.
  CleanMemory();
  ResetSettings();

  // Set parameters using the same input but with a larger tolerance.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("optimizer", std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 0);
  SetInputParam("tolerance", (double) 100.0);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that output is different when batch_size is different.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentBatchSizeTest",
                "[NCAMainTest][BindingTests]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Simple dataset with 6 points and two classes.
  arma::mat x              = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                             " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels = " 0    0    0    1    1    1   ";

  // Set parameters with a small batch_size.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("batch_size", (int) 2);
  SetInputParam("linear_scan", true);
  SetInputParam("max_iterations", 1000);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset settings.
  CleanMemory();
  ResetSettings();

  // Input the same dataset.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but with a larger batch_size.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("batch_size", (int) 3);
  SetInputParam("linear_scan", true);
  SetInputParam("max_iterations", 1000);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that output is different when setting linear_scan to true.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALinearScanTest",
                "[NCAMainTest][BindingTests]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Simple dataset with 6 points and two classes.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels  = " 0    0    0    1    1    1   ";

  // Set parameters.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", labels);
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("max_iterations", 100);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset settings.
  CleanMemory();
  ResetSettings();

  // Input the same dataset.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  // Set parameters using the same input but rely on linear_scan flag
  // default value false.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", labels2);
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("max_iterations", 100);
  SetInputParam("linear_scan", true);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that output is the same when same data is used with linear_scan set.
 */
TEST_CASE_METHOD(NCATestFixture, "NCALinearScanTest2",
                "[NCAMainTest][BindingTests]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Simple dataset with 6 points and two classes.
  arma::mat x               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels  = " 0    0    0    1    1    1   ";

  // Set same parameter with same data.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", labels);
  SetInputParam("linear_scan", true);
  SetInputParam("max_iterations", 1000);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset Settings.
  CleanMemory();
  ResetSettings();

  // Set same parameter using the same data.
  arma::mat y               = "-0.1 -0.1 -0.1  0.1  0.1  0.1;"
                              " 1.0  0.0 -1.0  1.0  0.0 -1.0 ";
  arma::Row<size_t> labels2 = " 0    0    0    1    1    1   ";

  SetInputParam("input", std::move(y));
  SetInputParam("labels", labels2);
  SetInputParam("linear_scan", true);
  SetInputParam("max_iterations", 1000);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are equal.
  CheckMatrices(output, params.Get<arma::mat>("output"));

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that different value of NumBasis results in a
 * different output matrix.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentNumBasisTest",
                "[NCAMainTest][BindingTests]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Simple dataset.
  arma::mat x;
  x.randu(4, 100);
  arma::Row<size_t> labels = arma::randi<arma::Row<size_t>>(100,
      DistrParam(0, 1));

  arma::mat y = x;
  arma::Row<size_t> labels2 = labels;

  // Set parameters and use a larger num_basis.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_basis", (int) 5);
  SetInputParam("max_iterations", (int) 10);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset Settings.
  CleanMemory();
  ResetSettings();

  // Set parameters with a smaller num_basis.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("num_basis", (int) 1);
  SetInputParam("max_iterations", (int) 10);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}

/**
 * Ensure that using a different value of max_iteration
 * results in a different output matrix.
 */
TEST_CASE_METHOD(NCATestFixture, "NCADifferentMaxIterationTest",
                "[NCAMainTest][BindingTests][long]")
{
  #if defined(MLPACK_USE_OPENMP)
  const size_t oldThreads = omp_get_num_threads();
  omp_set_num_threads(1);
  #endif

  // Random dataset.
  arma::mat x;
  x.randu(3, 600);
  arma::Row<size_t> labels = arma::randi<arma::Row<size_t>>(600,
      DistrParam(0, 1));

  arma::mat y = x;
  arma::Row<size_t> labels2 = labels;

  // Set parameters with a small max_iterations.
  SetInputParam("input", std::move(x));
  SetInputParam("labels", std::move(labels));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 3);
  FixedRandomSeed();

  RUN_BINDING();

  arma::mat output = params.Get<arma::mat>("output");

  // Reset settings.
  CleanMemory();
  ResetSettings();

  // Set parameters using the same input but with a larger max_iterations.
  SetInputParam("input", std::move(y));
  SetInputParam("labels", std::move(labels2));
  SetInputParam("optimizer",  std::string("lbfgs"));
  SetInputParam("max_iterations", (int) 20);
  FixedRandomSeed();

  RUN_BINDING();

  // Check that the output matrices are different.
  REQUIRE(accu(params.Get<arma::mat>("output") != output) > 0);

  #if defined(MLPACK_USE_OPENMP)
  omp_set_num_threads(oldThreads);
  #endif
}
