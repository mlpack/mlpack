/**
 * @file tests/main_tests/fastmks_test.cpp
 * @author Yashwant Singh
 * @author Prabhat Sharma
 *
 * Test RUN_BINDING() of fastmks_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/fastmks/fastmks_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(FastMKSTestFixture);

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSEqualDimensionTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points in query and reference matrices
  // are allowed to be different
  // 90 points in 2 dimensions.
  arma::mat queryData(2, 90, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 4);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/*
 * Check that we can't specify an invalid k when only reference
 * matrix is given.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSInvalidKTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 50 points in 3 dimensions.
  arma::mat referenceData(3, 50, arma::fill::randu);

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 51); // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/**
 * Check that when k is specified, it must be greater than 0.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSZeroKTest",
                 "[FastMKSMainTest][BindingTests]")
{
  arma::mat referenceData(3, 50, arma::fill::randu);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 0); // Invalid when reference is specified.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSInvalidKQueryDataTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 50 points in 3 dimensions.
  arma::mat referenceData(3, 50, arma::fill::randu);
  // 10 points in 3 dimensions.
  arma::mat queryData(3, 10, arma::fill::randu);

  // Random input, some k > number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 51);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::invalid_argument);
}

/*
 * Check that we can't pass both input_model and reference matrix.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSRefModelTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  FastMKSModel* m = params.Get<FastMKSModel*>("output_model");
  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  // Input pre-trained model.
  SetInputParam("input_model", m);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't pass an invalid kernel.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSInvalidKernelTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);
  string kernelName = "dummy";

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", std::move(kernelName)); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure that dimensions of the indices and kernel
 * matrices are correct given a value of k.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSOutputDimensionTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  // Check the indices matrix has 10 points for each input point.
  REQUIRE(params.Get<arma::Mat<size_t>>("indices").n_rows == 10);
  REQUIRE(params.Get<arma::Mat<size_t>>("indices").n_cols == 100);

  // Check the kernel matrix has 10 points for each input point.
  REQUIRE(params.Get<arma::mat>("kernels").n_rows == 10);
  REQUIRE(params.Get<arma::mat>("kernels").n_cols == 100);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSModelReuseTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);
  // 90 points in 3 dimensions.
  arma::mat queryData(3, 90, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", queryData);

  RUN_BINDING();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  FastMKSModel* output_model;
  indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  kernel = std::move(params.Get<arma::mat>("kernels"));
  output_model = std::move(params.Get<FastMKSModel*>("output_model"));

  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);

  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(indices, params.Get<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel, params.Get<arma::mat>("kernels"));
}

/*
 * Ensure that reference dataset gives the same result when passed as
 * a query dataset
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSQueryRefTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);
  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("query", referenceData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  kernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("query", referenceData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  CheckMatrices(indices, params.Get<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel, params.Get<arma::mat>("kernels"));
}

/*
 * Ensure that naive mode returns the same result as tree mode.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSNaiveModeTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  kernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("naive", true);

  RUN_BINDING();

  CheckMatrices(indices, params.Get<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel, params.Get<arma::mat>("kernels"));
}

/*
 * Ensure that single-tree search returns the same result as dual-tree search.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSTreeTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  RUN_BINDING();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  kernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("single", true);

  RUN_BINDING();

  CheckMatrices(indices, params.Get<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel, params.Get<arma::mat>("kernels"));
}

/*
 * Ensure that we get almost same results in cover tree search mode when
 * different basis is specified.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSBasisTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("base", 3.0);

  RUN_BINDING();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  kernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("base", 4.0);

  RUN_BINDING();

  arma::Mat<size_t> newindices;
  arma::mat newkernel;
  newindices = std::move(params.Get<arma::Mat<size_t>>("indices"));
  newkernel = std::move(params.Get<arma::mat>("kernels"));

  CheckMatrices(indices, newindices);
  CheckMatrices(kernel, newkernel);
}

/**
 * Check that we can't specify base less than 1.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSBaseTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, invalid base.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("base", 0.0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure that different kernels returns different results.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSKernelTest",
                 "[FastMKSMainTest][BindingTests]")
{
  std::string kerneltypes[] = {"polynomial", "cosine", "gaussian",
      "epanechnikov", "triangular", "hyptan"};
  const int nofkerneltypes = 6;
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);
  // 90 points in 3 dimensions.
  arma::mat queryData(3, 90, arma::fill::randu);
  // Keep some k <= number of reference points same over all.
  SetInputParam("k", (int) 10);
  // For Hyptan Kernel
  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Cannot load test dataset data_3d_ind.txt!");

  arma::Mat<size_t> indicesCompare;
  arma::mat kernelsCompare;

  arma::Mat<size_t> indices;
  arma::mat kernels;

  // Looping over all the kernels
  for (size_t i = 0; i < nofkerneltypes; ++i)
  {
    if (kerneltypes[i] == "hyptan")
    {
      // Same random inputs, different algorithms.
      SetInputParam("reference", inputData);
      SetInputParam("query", inputData);
      SetInputParam("kernel", kerneltypes[i]);
    }
    else
    {
      // Same random inputs, different algorithms.
      SetInputParam("reference", referenceData);
      SetInputParam("query", queryData);
      SetInputParam("kernel", kerneltypes[i]);
    }
    RUN_BINDING();

    if (i == 0)
    {
      indicesCompare =
         std::move(params.Get<arma::Mat<size_t>>("indices"));
      kernelsCompare = std::move(params.Get<arma::mat>("kernels"));
    }
    else
    {
      indices = std::move(params.Get<arma::Mat<size_t>>("indices"));
      kernels = std::move(params.Get<arma::mat>("kernels"));

      CheckMatricesNotEqual(indicesCompare, indices);
      CheckMatricesNotEqual(kernelsCompare, kernels);
    }

    // Reset passed parameters.
    ResetSettings();

    if (i != nofkerneltypes - 1)
      CleanMemory();
  }
}

/**
 * Ensure that offset affects the final result of polynomial and hyptan kernel.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSOffsetTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);
  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "polynomial");
  SetInputParam("offset", 1.0);

  RUN_BINDING();

  arma::mat polyKernel;
  polyKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "polynomial");
  SetInputParam("offset", 4.0);

  RUN_BINDING();

  CheckMatricesNotEqual(polyKernel, params.Get<arma::mat>("kernels"));

  CleanMemory();

  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Cannot load test dataset data_3d_ind.txt!");

  ResetSettings();

  SetInputParam("reference", inputData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (std::string) "hyptan");
  SetInputParam("offset", 1.0);

  RUN_BINDING();

  arma::mat hyptanKernel;
  hyptanKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", inputData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (std::string) "hyptan");
  SetInputParam("offset", 4.0);
  RUN_BINDING();

  CheckMatricesNotEqual(hyptanKernel, params.Get<arma::mat>("kernels"));
}

/**
 * Ensure that degree affects the final result of polynomial kernel.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSDegreeTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);
  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "polynomial");
  SetInputParam("degree", 2.0); // Default value.

  RUN_BINDING();

  arma::mat polyKernel;
  polyKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "polynomial");
  SetInputParam("degree", 4.0);

  RUN_BINDING();

  CheckMatricesNotEqual(polyKernel, params.Get<arma::mat>("kernels"));
}

/**
 * Ensure that scale affects the final result of hyptan kernel.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSScaleTest",
                 "[FastMKSMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Cannot load test dataset data_3d_ind.txt!");

  // Random input, some k <= number of reference points.
  SetInputParam("reference", inputData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (std::string) "hyptan");
  SetInputParam("scale", 1.0); // Default value.

  RUN_BINDING();

  arma::mat hyptanKernel;
  hyptanKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", inputData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (std::string) "hyptan");
  SetInputParam("scale", 1.5);

  RUN_BINDING();

  CheckMatricesNotEqual(hyptanKernel, params.Get<arma::mat>("kernels"));
}

/**
 * Ensure that bandwidth affects the final result of Gaussian, Epanechnikov, and
 * triangular kernel.
 */
TEST_CASE_METHOD(FastMKSTestFixture, "FastMKSBandwidthTest",
                 "[FastMKSMainTest][BindingTests]")
{
  // 100 points in 3 dimensions.
  arma::mat referenceData(3, 100, arma::fill::randu);

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "gaussian");
  SetInputParam("bandwidth", 1.0); // Default value.

  RUN_BINDING();

  arma::mat gaussianKernel;
  gaussianKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "gaussian");
  SetInputParam("bandwidth", 4.0);

  RUN_BINDING();
  CheckMatricesNotEqual(gaussianKernel, params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "epanechnikov");
  SetInputParam("bandwidth", 1.0); // Default value.

  RUN_BINDING();

  arma::mat epanKernel;
  epanKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "epanechnikov");
  SetInputParam("bandwidth", 4.0);

  RUN_BINDING();
  CheckMatricesNotEqual(epanKernel, params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "triangular");
  SetInputParam("bandwidth", 1.0); // Default value.

  RUN_BINDING();

  arma::mat triKernel;
  triKernel = std::move(params.Get<arma::mat>("kernels"));

  CleanMemory();
  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", (string) "triangular");
  SetInputParam("bandwidth", 4.0);

  RUN_BINDING();

  CheckMatricesNotEqual(triKernel, params.Get<arma::mat>("kernels"));
}
