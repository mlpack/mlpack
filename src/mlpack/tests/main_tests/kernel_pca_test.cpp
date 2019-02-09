/**
 * @file kernel_pca_test.cpp
 * @author Saksham Bansal
 *
 * Test mlpackMain() of kernel_pca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "KernelPrincipalComponentsAnalysis";

#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/kernel_pca/kernel_pca_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KernelPCATestFixture
{
 public:
  KernelPCATestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~KernelPCATestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

static void ResetSettings()
{
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(KernelPCAMainTest, KernelPCATestFixture);

/**
 * Make sure that all valid kernels return correct output dimension.
 */
BOOST_AUTO_TEST_CASE(KernelPCADimensionTest)
{
  std::string kernels[] = {
      "linear", "gaussian", "polynomial",
      "hyptan", "laplacian", "epanechnikov", "cosine"
  };

  for (std::string& kernel : kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);
    // Random input, new dimensionality of 3.
    SetInputParam("input", std::move(x));
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    mlpackMain();

    // Now check that the output has 3 dimensions.
    BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
    BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
  }
}

/**
 * Check that error is thrown when no kernel is specified.
 */
BOOST_AUTO_TEST_CASE(KernelPCANoKernelTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that error is thrown when an invalid kernel is specified.
 */
BOOST_AUTO_TEST_CASE(KernelPCAInvalidKernelTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "badName");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure for zero dimensionality, we get a dataset with the same dimensionality
 * as the input dataset.
 */
BOOST_AUTO_TEST_CASE(KernelPCA0DimensionalityTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 0);
  SetInputParam("kernel", (std::string) "gaussian");
  mlpackMain();

  // Now check that the output has same dimensions as input.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 5);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
}

/**
 * Make sure that centering the dataset makes a difference.
 */
BOOST_AUTO_TEST_CASE(KernelPCACenterTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  // Get output without centering the dataset.
  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "linear");
  mlpackMain();
  arma::mat output1 = CLI::GetParam<arma::mat>("output");

  // Get output after centering the dataset.
  SetInputParam("input", std::move(x));
  SetInputParam("center", true);
  mlpackMain();
  arma::mat output2 = CLI::GetParam<arma::mat>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

/**
 * Check that we can't specify an invalid new dimensionality.
 */
BOOST_AUTO_TEST_CASE(KernelPCATooHighNewDimensionalityTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 7); // Invalid.
  SetInputParam("kernel", (std::string) "linear");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that error is thrown when no input is specified.
 */
BOOST_AUTO_TEST_CASE(KernelPCANoInputTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("new_dimensionality", (int) 2);
  SetInputParam("kernel", (std::string) "linear");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that error is thrown if invalid sampling scheme is specified.
 */
BOOST_AUTO_TEST_CASE(KernelPCABadSamplingTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "linear");
  SetInputParam("nystroem_method", true);
  SetInputParam("sampling", (std::string) "badName");

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Test that bandwidth effects the result for gaussian, epanechnikov
 * and laplacian kernels.
 */
BOOST_AUTO_TEST_CASE(KernelPCABandWidthTest)
{
  std::string kernels[] = {
      "gaussian", "epanechnikov", "laplacian"
  };

  for (std::string& kernel : kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 5);

    // Get output using bandwidth 1.
    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    SetInputParam("bandwidth", (double) 1);

    mlpackMain();
    arma::mat output1 = CLI::GetParam<arma::mat>("output");

    // Get output using bandwidth 2.
    SetInputParam("input", std::move(x));
    SetInputParam("bandwidth", (double) 2);

    mlpackMain();
    arma::mat output2 = CLI::GetParam<arma::mat>("output");

    // The resulting matrices should be different.
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }
}

/**
 * Test that offset effects the result for polynomial and hyptan kernels.
 */
BOOST_AUTO_TEST_CASE(KernelPCAOffsetTest)
{
  std::string kernels[] = {
      "polynomial", "hyptan"
  };

  for (std::string& kernel : kernels)
  {
    ResetSettings();
    arma::mat x = arma::randu<arma::mat>(5, 100);

    SetInputParam("input", x);
    SetInputParam("new_dimensionality", (int) 3);
    SetInputParam("kernel", kernel);
    SetInputParam("offset", (double) 0.01);

    mlpackMain();
    arma::mat output1 = CLI::GetParam<arma::mat>("output");

    SetInputParam("input", std::move(x));
    SetInputParam("offset", (double) 0.1);

    mlpackMain();
    arma::mat output2 = CLI::GetParam<arma::mat>("output");

    // The resulting matrices should be different.
    BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  }
}

/**
 * Test that degree effects the result for polynomial kernel.
 */
BOOST_AUTO_TEST_CASE(KernelPCADegreeTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "polynomial");
  SetInputParam("degree", (double) 2);

  mlpackMain();
  arma::mat output1 = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", std::move(x));
  SetInputParam("degree", (double) 3);

  mlpackMain();
  arma::mat output2 = CLI::GetParam<arma::mat>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

/**
 * Test that kernel scale effects the result for hyptan kernel.
 */
BOOST_AUTO_TEST_CASE(KernelPCAKernelScaleTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "hyptan");
  SetInputParam("kernel_scale", (double) 2);

  mlpackMain();
  arma::mat output1 = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", std::move(x));
  SetInputParam("kernel_scale", (double) 3);

  mlpackMain();
  arma::mat output2 = CLI::GetParam<arma::mat>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

/**
 * Test that using a sampling scheme with nystroem method makes a difference.
 */
BOOST_AUTO_TEST_CASE(KernelPCASamplingSchemeTest)
{
  ResetSettings();

  arma::mat x = arma::randu<arma::mat>(5, 500);

  SetInputParam("input", x);
  SetInputParam("new_dimensionality", (int) 3);
  SetInputParam("kernel", (std::string) "gaussian");
  SetInputParam("nystroem_method", true);
  SetInputParam("sampling", (std::string) "kmeans");

  mlpackMain();

  arma::mat output1 = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", x);
  SetInputParam("sampling", (std::string) "random");

  mlpackMain();
  arma::mat output2 = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", x);
  SetInputParam("sampling", (std::string) "ordered");

  mlpackMain();
  arma::mat output3 = CLI::GetParam<arma::mat>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
  BOOST_REQUIRE(arma::any(arma::vectorise(output2 != output3)));
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output3)));
}

BOOST_AUTO_TEST_SUITE_END();
