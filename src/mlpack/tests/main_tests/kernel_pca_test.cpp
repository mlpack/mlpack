/**
 * @file tests/main_tests/kernel_pca_test.cpp
 * @author Saksham Bansal
 *
 * Test RUN_BINDING() of kernel_pca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/methods/kernel_pca/kernel_pca_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(KernelPCATestFixture);

/**
 * Make sure that all valid kernels return correct output dimension.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCADimensionTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  std::string kernels[] = {
      "linear", "gaussian", "polynomial",
      "hyptan", "laplacian", "epanechnikov", "cosine"
  };

  for (std::string& kernel : kernels)
  {
    CleanMemory();
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);
    // Random input, new dimensionality of 3.
    SetInputParam("input", std::move(x));
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    RUN_BINDING();

    // Now check that the output has 3 dimensions.
    REQUIRE(params.Get<arma::mat>("output").n_rows == 3);
    REQUIRE(params.Get<arma::mat>("output").n_cols == 5);
  }
}

/**
 * Check that error is thrown when no kernel is specified.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCANoKernelTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that error is thrown when an invalid kernel is specified.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCAInvalidKernelTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "badName");

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensure for zero dimensionality, we get a dataset with the same dimensionality
 * as the input dataset.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCA0DimensionalityTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 0);
  SetInputParam("kernel", (std::string) "gaussian");
  RUN_BINDING();

  // Now check that the output has same dimensions as input.
  REQUIRE(params.Get<arma::mat>("output").n_rows == 5);
  REQUIRE(params.Get<arma::mat>("output").n_cols == 5);
}

/**
 * Make sure that centering the dataset makes a difference.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCACenterTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  // Get output without centering the dataset.
  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "linear");
  RUN_BINDING();
  arma::mat output1 = params.Get<arma::mat>("output");

  // Get output after centering the dataset.
  SetInputParam("input", std::move(x));
  SetInputParam("center", true);
  RUN_BINDING();
  arma::mat output2 = params.Get<arma::mat>("output");

  // The resulting matrices should be different.
  REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

/**
 * Check that we can't specify an invalid new dimensionality.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCATooHighNewDimensionalityTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 7); // Invalid.
  SetInputParam("kernel", (std::string) "linear");

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that error is thrown when no input is specified.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCANoInputTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("new_dimensionality", (int) 2);
  SetInputParam("kernel", (std::string) "linear");

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that error is thrown if invalid sampling scheme is specified.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCABadSamplingTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "linear");
  SetInputParam("nystroem_method", true);
  SetInputParam("sampling", (std::string) "badName");

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Test that bandwidth effects the result for gaussian, epanechnikov
 * and laplacian kernels.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCABandWidthTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  std::string kernels[] = {
      "gaussian", "epanechnikov", "laplacian"
  };

  for (std::string& kernel : kernels)
  {
    CleanMemory();
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);

    // Get output using bandwidth 1.
    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    SetInputParam("bandwidth", (double) 1);

    RUN_BINDING();
    arma::mat output1 = params.Get<arma::mat>("output");

    // Get output using bandwidth 2.
    SetInputParam("input", std::move(x));
    SetInputParam("bandwidth", (double) 2);

    RUN_BINDING();
    arma::mat output2 = params.Get<arma::mat>("output");

    // The resulting matrices should be different.
    REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }
}

/**
 * Test that offset effects the result for polynomial and hyptan kernels.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCAOffsetTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  std::string kernels[] = {
      "polynomial", "hyptan"
  };

  for (std::string& kernel : kernels)
  {
    CleanMemory();
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 100);

    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    SetInputParam("offset", (double) 0.01);

    RUN_BINDING();
    arma::mat output1 = params.Get<arma::mat>("output");

    SetInputParam("input", std::move(x));
    SetInputParam("offset", (double) 0.1);

    RUN_BINDING();
    arma::mat output2 = params.Get<arma::mat>("output");

    // The resulting matrices should be different.
    REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }
}

/**
 * Test that degree effects the result for polynomial kernel.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCADegreeTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "polynomial");
  SetInputParam("degree", (double) 2);

  RUN_BINDING();
  arma::mat output1 = params.Get<arma::mat>("output");

  SetInputParam("input", std::move(x));
  SetInputParam("degree", (double) 3);

  RUN_BINDING();
  arma::mat output2 = params.Get<arma::mat>("output");

  // The resulting matrices should be different.
  REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

/**
 * Test that kernel scale effects the result for hyptan kernel.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCAKernelScaleTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "hyptan");
  SetInputParam("kernel_scale", (double) 2);

  RUN_BINDING();
  arma::mat output1 = params.Get<arma::mat>("output");

  SetInputParam("input", std::move(x));
  SetInputParam("kernel_scale", (double) 3);

  RUN_BINDING();
  arma::mat output2 = params.Get<arma::mat>("output");

  // The resulting matrices should be different.
  REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

/**
 * Test that using a sampling scheme with nystroem method makes a difference.
 */
TEST_CASE_METHOD(KernelPCATestFixture, "KernelPCASamplingSchemeTest",
                 "[KernelPCAMainTest][BindingTests]")
{
  CleanMemory();
  ResetSettings();

  arma::mat x = arma::randu<arma::mat>(5, 500);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "gaussian");
  SetInputParam("nystroem_method", true);
  SetInputParam("sampling", (std::string) "kmeans");

  RUN_BINDING();

  arma::mat output1 = params.Get<arma::mat>("output");

  SetInputParam("input", x);
  SetInputParam("sampling", (std::string) "random");

  RUN_BINDING();
  arma::mat output2 = params.Get<arma::mat>("output");

  SetInputParam("input", x);
  SetInputParam("sampling", (std::string) "ordered");

  RUN_BINDING();
  arma::mat output3 = params.Get<arma::mat>("output");

  // The resulting matrices should be different.
  REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  REQUIRE(arma::any(arma::vectorise(output2 != output3)));
  REQUIRE(arma::any(arma::vectorise(output1 != output3)));
}
