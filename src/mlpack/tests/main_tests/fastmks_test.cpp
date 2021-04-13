/**
 * @file tests/main_tests/fastmks_test.cpp
 * @author Yashwant Singh
 * @author Prabhat Sharma
 *
 * Test mlpackMain() of fastmks_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "FastMaxKernelSearch";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/fastmks/fastmks_main.cpp>

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

struct FastMKSTestFixture
{
 public:
  FastMKSTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~FastMKSTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
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

  mlpackMain();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  SetInputParam("reference", std::move(referenceData));
  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(IO::GetParam<FastMKSModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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

  mlpackMain();

  // Check the indices matrix has 10 points for each input point.
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("indices").n_rows == 10);
  REQUIRE(IO::GetParam<arma::Mat<size_t>>("indices").n_cols == 100);

  // Check the kernel matrix has 10 points for each input point.
  REQUIRE(IO::GetParam<arma::mat>("kernels").n_rows == 10);
  REQUIRE(IO::GetParam<arma::mat>("kernels").n_cols == 100);
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

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  FastMKSModel* output_model;
  indices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(IO::GetParam<arma::mat>("kernels"));
  output_model = std::move(IO::GetParam<FastMKSModel*>("output_model"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(indices, IO::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel, IO::GetParam<arma::mat>("kernels"));
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

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["query"].wasPassed = false;


  SetInputParam("reference", referenceData);
  SetInputParam("query", referenceData);

  mlpackMain();

  CheckMatrices(indices,
      IO::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel,
      IO::GetParam<arma::mat>("kernels"));
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

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["k"].wasPassed = false;

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("naive", true);

  mlpackMain();

  CheckMatrices(indices,
      IO::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel,
      IO::GetParam<arma::mat>("kernels"));
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

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("single", true);

  mlpackMain();

  CheckMatrices(indices,
      IO::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel,
      IO::GetParam<arma::mat>("kernels"));
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

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("base", 4.0);

  mlpackMain();

  arma::Mat<size_t> newindices;
  arma::mat newkernel;
  newindices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
  newkernel = std::move(IO::GetParam<arma::mat>("kernels"));

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

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
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
    mlpackMain();

    if (i == 0)
    {
      indicesCompare =
         std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
      kernelsCompare = std::move(IO::GetParam<arma::mat>("kernels"));
    }
    else
    {
      indices = std::move(IO::GetParam<arma::Mat<size_t>>("indices"));
      kernels = std::move(IO::GetParam<arma::mat>("kernels"));

      CheckMatricesNotEqual(indicesCompare, indices);
      CheckMatricesNotEqual(kernelsCompare, kernels);
    }

    // Reset passed parameters.
    IO::GetSingleton().Parameters()["reference"].wasPassed = false;
    IO::GetSingleton().Parameters()["query"].wasPassed = false;
    IO::GetSingleton().Parameters()["kernel"].wasPassed = false;

    if (i != nofkerneltypes - 1)
      bindings::tests::CleanMemory();
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
  SetInputParam("kernel", (string)"polynomial");
  SetInputParam("offset", 1.0);

  mlpackMain();

  arma::mat polyKernel;
  polyKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["offset"].wasPassed = false;
  IO::GetParam<FastMKSModel*>("input_model") = NULL;
  IO::GetParam<FastMKSModel*>("output_model") = NULL;

  SetInputParam("reference", referenceData);
  SetInputParam("offset", 4.0);

  mlpackMain();

  CheckMatricesNotEqual(polyKernel,
      IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Cannot load test dataset data_3d_ind.txt!");

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["kernel"].wasPassed = false;
  IO::GetSingleton().Parameters()["offset"].wasPassed = false;
  IO::GetParam<FastMKSModel*>("input_model") = NULL;
  IO::GetParam<FastMKSModel*>("output_model") = NULL;

  SetInputParam("reference", inputData);
  SetInputParam("kernel", (std::string)"hyptan");
  SetInputParam("offset", 1.0);

  mlpackMain();

  arma::mat hyptanKernel;
  hyptanKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["offset"].wasPassed = false;
  IO::GetParam<FastMKSModel*>("input_model") = NULL;
  IO::GetParam<FastMKSModel*>("output_model") = NULL;

  SetInputParam("reference", inputData);
  SetInputParam("offset", 4.0);
  mlpackMain();

  CheckMatricesNotEqual(hyptanKernel,
      IO::GetParam<arma::mat>("kernels"));
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
  SetInputParam("kernel", (string)"polynomial");
  SetInputParam("degree", 2.0); // Default value.

  mlpackMain();

  arma::mat polyKernel;
  polyKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["degree"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("degree", 4.0);

  mlpackMain();

  CheckMatricesNotEqual(polyKernel,
      IO::GetParam<arma::mat>("kernels"));
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
  SetInputParam("kernel", (std::string)"hyptan");
  SetInputParam("scale", 1.0); // Default value.

  mlpackMain();

  arma::mat hyptanKernel;
  hyptanKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["scale"].wasPassed = false;

  SetInputParam("reference", inputData);
  SetInputParam("scale", 1.5);

  mlpackMain();

  CheckMatricesNotEqual(hyptanKernel,
      IO::GetParam<arma::mat>("kernels"));
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
  SetInputParam("kernel", (string)"gaussian");
  SetInputParam("bandwidth", 1.0); // Default value.

  mlpackMain();

  arma::mat gaussianKernel;
  gaussianKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["bandwidth"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("bandwidth", 4.0);

  mlpackMain();
  CheckMatricesNotEqual(gaussianKernel,
      IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["bandwidth"].wasPassed = false;
  IO::GetSingleton().Parameters()["kernel"].wasPassed = false;

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("kernel", (string)"epanechnikov");
  SetInputParam("bandwidth", 1.0); // Default value.

  mlpackMain();

  arma::mat epanKernel;
  epanKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["bandwidth"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("bandwidth", 4.0);

  mlpackMain();
  CheckMatricesNotEqual(epanKernel,
       IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["bandwidth"].wasPassed = false;
  IO::GetSingleton().Parameters()["kernel"].wasPassed = false;

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("kernel", (string)"triangular");
  SetInputParam("bandwidth", 1.0); // Default value.

  mlpackMain();

  arma::mat triKernel;
  triKernel = std::move(IO::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  IO::GetSingleton().Parameters()["reference"].wasPassed = false;
  IO::GetSingleton().Parameters()["bandwidth"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("bandwidth", 4.0);

  mlpackMain();

  CheckMatricesNotEqual(triKernel,
      IO::GetParam<arma::mat>("kernels"));
}
