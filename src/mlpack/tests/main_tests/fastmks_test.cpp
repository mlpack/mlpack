/**
 * @file fastmks_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct FMKSTestFixture
{
 public:
  FMKSTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~FMKSTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(FMKSMainTest, FMKSTestFixture);

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
BOOST_AUTO_TEST_CASE(FMKSEqualDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points in query and reference matrices
  // are allowed to be different
  arma::mat queryData;
  queryData.randu(2, 90); // 90 points in 2 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 4);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when only reference
 * matrix is given.
 */
BOOST_AUTO_TEST_CASE(FMKSInvalidKTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 50); // 50 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 51);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::bad_alloc);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
BOOST_AUTO_TEST_CASE(FMKSInvalidKQueryDataTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 50); // 50 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 10); // 10 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 51);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::invalid_argument);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass both input_model and reference matrix.
 */
BOOST_AUTO_TEST_CASE(FMKSRefModelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(CLI::GetParam<FastMKSModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid kernel.
 */
BOOST_AUTO_TEST_CASE(FMKSInvalidKernelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.
  string kernelName = "dummy";

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", std::move(kernelName)); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that dimensions of the indices and kernel
 * matrices are correct given a value of k.
 */
BOOST_AUTO_TEST_CASE(FMKSOutputDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);

  mlpackMain();

  // Check the indices matrix has 10 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>
      ("indices").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>
      ("indices").n_cols, 100);

  // Check the kernel matrix has 10 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("kernels").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("kernels").n_cols, 100);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(FMKSModelReuseTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  FastMKSModel* output_model;
  indices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(CLI::GetParam<arma::mat>("kernels"));
  output_model = std::move(CLI::GetParam<FastMKSModel*>("output_model"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);

  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(indices, CLI::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel, CLI::GetParam<arma::mat>("kernels"));
}

/*
 * Ensure that reference dataset gives the same result when passed as
 * a query dataset
 */
BOOST_AUTO_TEST_CASE(FMKSQueryRefTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("query", referenceData);

  mlpackMain();

  CheckMatrices(indices,
      CLI::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel,
      CLI::GetParam<arma::mat>("kernels"));
}

/*
 * Ensure that naive mode returns the same result as tree mode.
 */
BOOST_AUTO_TEST_CASE(FMKSNaiveModeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("naive", true);

  mlpackMain();

  CheckMatrices(indices,
      CLI::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel,
      CLI::GetParam<arma::mat>("kernels"));
}

/*
 * Ensure that single-tree search returns the same result
 * as dual-tree search
 */
BOOST_AUTO_TEST_CASE(FMKSTreeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("single", true);

  mlpackMain();

  CheckMatrices(indices,
      CLI::GetParam<arma::Mat<size_t>>("indices"));
  CheckMatrices(kernel,
      CLI::GetParam<arma::mat>("kernels"));
}

/*
 * Ensure that we get almost same results in cover tree
 * search mode when different basis is specified.
 */
BOOST_AUTO_TEST_CASE(FMKSBasisTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("base", 3.0);

  mlpackMain();

  arma::Mat<size_t> indices;
  arma::mat kernel;
  indices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  kernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 10);
  SetInputParam("base", 4.0);

  mlpackMain();

  arma::Mat<size_t> newindices;
  arma::mat newkernel;
  newindices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  newkernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  CheckMatrices(indices, newindices);
  CheckMatrices(kernel, newkernel);
}

/*
 * Ensure that different kernels returns different results.
 */
/*
BOOST_AUTO_TEST_CASE(FMKSKernelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  // Default linear kernel

  mlpackMain();

  arma::Mat<size_t> linearIndices;
  arma::mat linearKernel;
  linearIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  linearKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName1 = "polynomial";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName1);

  mlpackMain();

  arma::Mat<size_t> polyIndices;
  arma::mat polyKernel;
  polyIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  polyKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName2 = "cosine";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName2);

  mlpackMain();

  arma::Mat<size_t> cosineIndices;
  arma::mat cosineKernel;
  cosineIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  cosineKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName3 = "gaussian";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName3);

  mlpackMain();

  arma::Mat<size_t> gaussianIndices;
  arma::mat gaussianKernel;
  gaussianIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  gaussianKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName4 = "epanechnikov";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName4);

  mlpackMain();

  arma::Mat<size_t> epaneIndices;
  arma::mat epaneKernel;
  epaneIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  epaneKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName5 = "triangular";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName5);

  mlpackMain();

  arma::Mat<size_t> triIndices;
  arma::mat triKernel;
  triIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  triKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName6 = "hyptan";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName6);

  mlpackMain();

  arma::Mat<size_t> hyptanIndices;
  arma::mat hyptanKernel;
  hyptanIndices = std::move(CLI::GetParam<arma::Mat<size_t>>("indices"));
  hyptanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  BOOST_REQUIRE_GT(arma::accu(linearIndices != polyIndices), 1);
  BOOST_REQUIRE_GT(arma::accu(polyIndices != cosineIndices), 1);
  BOOST_REQUIRE_GT(arma::accu(cosineIndices != gaussianIndices), 1);
  BOOST_REQUIRE_GT(arma::accu(gaussianIndices != epaneIndices), 1);
  BOOST_REQUIRE_GT(arma::accu(epaneIndices != hyptanIndices), 1);

  BOOST_REQUIRE_GT(arma::accu(linearKernel != polyKernel), 1);
  BOOST_REQUIRE_GT(arma::accu(polyKernel != cosineKernel), 1);
  BOOST_REQUIRE_GT(arma::accu(cosineKernel != gaussianKernel), 1);
  BOOST_REQUIRE_GT(arma::accu(gaussianKernel != epaneKernel), 1);
  BOOST_REQUIRE_GT(arma::accu(epaneKernel != hyptanKernel), 1);
}
/** Ensure that offset affects the final result of polynomial
 * and hyptan kernel
 */
/*
BOOST_AUTO_TEST_CASE(FMKSOffsetTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

	string kernelName1 = "polynomial";

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", kernelName1);
  SetInputParam("offset", 1.0);

  mlpackMain();

  arma::mat polyKernel;
  polyKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["offset"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("offset", 4.0);

  arma::mat newpolyKernel;
  newpolyKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  BOOST_REQUIRE_GT(arma::accu(polyKernel != newpolyKernel), 1);

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;
  CLI::GetSingleton().Parameters()["offset"].wasPassed = false;

	string kernelName2 = "hyptan";

  SetInputParam("reference", referenceData);
  SetInputParam("kernel", kernelName2);
  SetInputParam("offset", 1.0);

  mlpackMain();

  arma::mat hyptanKernel;
  hyptanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["offset"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("offset", 4.0);

  arma::mat newhyptanKernel;
  newhyptanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  BOOST_REQUIRE_GT(arma::accu(hyptanKernel != newhyptanKernel), 1);
}*/

/** Ensure that degree affects the final result of polynomial kernel
 */
/*
BOOST_AUTO_TEST_CASE(FMKSDegreeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

	string kernelName1 = "polynomial";

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", kernelName1);
  SetInputParam("degree", 2.0); // Default value.

  mlpackMain();

  arma::mat polyKernel;
  polyKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["degree"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("degree", 4.0);

  arma::mat newpolyKernel;
  newpolyKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  BOOST_REQUIRE_GT(arma::accu(polyKernel != newpolyKernel), 1);
}*/

/** Ensure that scale affects the final result of hyptan kernel
 */
/*
BOOST_AUTO_TEST_CASE(FMKSScaleTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

	string kernelName1 = "hyptan";

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", kernelName1);
  SetInputParam("scale", 1.0); // Default value.

  mlpackMain();

  arma::mat hyptanKernel;
  hyptanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["scale"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("scale", 4.0);

  arma::mat newhyptanKernel;
  newhyptanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  BOOST_REQUIRE_GT(arma::accu(hyptanKernel != newhyptanKernel), 1);
}*/

/** Ensure that bandwidth affects the final result of Gaussian, Epanechnikov,
 *  and triangular kernel
 */
 /*
BOOST_AUTO_TEST_CASE(FMKSBandwidthTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

	string kernelName1 = "gaussian";

  // Random input, some k <= number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 10);
  SetInputParam("kernel", kernelName1);
  SetInputParam("bandwidth", 1.0); // Default value.

  mlpackMain();

  arma::mat gaussianKernel;
  gaussianKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  bindings::tests::CleanMemory();

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["bandwidth"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("bandwidth", 4.0);

  arma::mat newgaussianKernel;
  newgaussianKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

  BOOST_REQUIRE_GT(arma::accu(gaussianKernel != newgaussianKernel), 1);

	bindings::tests::CleanMemory();

	CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
	CLI::GetSingleton().Parameters()["bandwidth"].wasPassed = false;
	CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName2 = "epanechnikov";

	// Random input, some k <= number of reference points.
	SetInputParam("reference", referenceData);
	SetInputParam("kernel", kernelName2);
	SetInputParam("bandwidth", 1.0); // Default value.

	mlpackMain();

	arma::mat epanKernel;
	epanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

	bindings::tests::CleanMemory();

	CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
	CLI::GetSingleton().Parameters()["scale"].wasPassed = false;

	SetInputParam("reference", referenceData);
	SetInputParam("bandwidth", 4.0);

	arma::mat newepanKernel;
	newepanKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

	BOOST_REQUIRE_GT(arma::accu(epanKernel != newepanKernel), 1);

	bindings::tests::CleanMemory();

	CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
	CLI::GetSingleton().Parameters()["bandwidth"].wasPassed = false;
	CLI::GetSingleton().Parameters()["kernel"].wasPassed = false;

	string kernelName3 = "triangular";

	// Random input, some k <= number of reference points.
	SetInputParam("reference", referenceData);
	SetInputParam("kernel", kernelName3);
	SetInputParam("triangular", 1.0); // Default value.

	mlpackMain();

	arma::mat triKernel;
	triKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

	bindings::tests::CleanMemory();

	CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
	CLI::GetSingleton().Parameters()["scale"].wasPassed = false;

	SetInputParam("reference", referenceData);
	SetInputParam("bandwidth", 4.0);

	arma::mat newtriKernel;
	newtriKernel = std::move(CLI::GetParam<arma::mat>("kernels"));

	BOOST_REQUIRE_GT(arma::accu(triKernel != newtriKernel), 1);
}*/
BOOST_AUTO_TEST_SUITE_END();
