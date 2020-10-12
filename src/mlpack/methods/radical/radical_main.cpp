/**
 * @file methods/radical/radical_main.cpp
 * @author Nishant Mehta
 *
 * Executable for RADICAL. RADICAL is Robust, Accurate, Direct ICA
 * aLgorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include "radical.hpp"

// Program Name.
BINDING_NAME("RADICAL");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of RADICAL, a method for independent component analysis "
    "(ICA).  Given a dataset, this can decompose the dataset into an unmixing "
    "matrix and an independent component matrix; this can be useful for "
    "preprocessing.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of RADICAL, a method for independent component analysis "
    "(ICA).  Assuming that we have an input matrix X, the goal is to find a "
    "square unmixing matrix W such that Y = W * X and the dimensions of Y are "
    "independent components.  If the algorithm is running particularly slowly, "
    "try reducing the number of replicates."
    "\n\n"
    "The input matrix to perform ICA on should be specified with the " +
    PRINT_PARAM_STRING("input") + " parameter.  The output matrix Y may be "
    "saved with the " + PRINT_PARAM_STRING("output_ic") + " output parameter, "
    "and the output unmixing matrix W may be saved with the " +
    PRINT_PARAM_STRING("output_unmixing") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, to perform ICA on the matrix " + PRINT_DATASET("X") + " with "
    "40 replicates, saving the independent components to " +
    PRINT_DATASET("ic") + ", the following command may be used: "
    "\n\n" +
    PRINT_CALL("radical", "input", "X", "replicates", 40, "output_ic", "ic"));

// See also...
BINDING_SEE_ALSO("Independent component analysis on Wikipedia",
        "https://en.wikipedia.org/wiki/Independent_component_analysis");
BINDING_SEE_ALSO("ICA using spacings estimates of entropy (pdf)",
        "http://www.jmlr.org/papers/volume4/learned-miller03a/"
        "learned-miller03a.pdf");
BINDING_SEE_ALSO("mlpack::radical::Radical C++ class documentation",
        "@doxygen/classmlpack_1_1radical_1_1Radical.html");

PARAM_MATRIX_IN_REQ("input", "Input dataset for ICA.", "i");

PARAM_MATRIX_OUT("output_ic", "Matrix to save independent components to.", "o");
PARAM_MATRIX_OUT("output_unmixing", "Matrix to save unmixing matrix to.", "u");

PARAM_DOUBLE_IN("noise_std_dev", "Standard deviation of Gaussian noise.", "n",
    0.175);
PARAM_INT_IN("replicates", "Number of Gaussian-perturbed replicates to use "
    "(per point) in Radical2D.", "r", 30);
PARAM_INT_IN("angles", "Number of angles to consider in brute-force search "
    "during Radical2D.", "a", 150);
PARAM_INT_IN("sweeps", "Number of sweeps; each sweep calls Radical2D once for "
    "each pair of dimensions.", "S", 0);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);
PARAM_FLAG("objective", "If set, an estimate of the final objective function "
    "is printed.", "O");

using namespace mlpack;
using namespace mlpack::radical;
using namespace mlpack::math;
using namespace mlpack::util;
using namespace std;
using namespace arma;

static void mlpackMain()
{
  // Set random seed.
  if (IO::GetParam<int>("seed") != 0)
    RandomSeed((size_t) IO::GetParam<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  RequireAtLeastOnePassed({ "output_ic", "output_unmixing" }, false, "no output"
      " will be saved");

  // Check validity of parameters.
  RequireParamValue<int>("replicates", [](int x) { return x > 0; }, true,
      "number of replicates must be positive");
  RequireParamValue<double>("noise_std_dev", [](double x) { return x >= 0.0; },
      true, "standard deviation of Gaussian noise must be greater than or equal"
      " to 0");
  RequireParamValue<int>("angles", [](int x) { return x > 0; }, true,
      "number of angles must be positive");
  RequireParamValue<int>("sweeps", [](int x) { return x >= 0; }, true,
      "number of sweeps must be 0 or greater");

  // Load the data.
  mat matX = std::move(IO::GetParam<mat>("input"));

  // Load parameters.
  double noiseStdDev = IO::GetParam<double>("noise_std_dev");
  size_t nReplicates = IO::GetParam<int>("replicates");
  size_t nAngles = IO::GetParam<int>("angles");
  size_t nSweeps = IO::GetParam<int>("sweeps");

  if (nSweeps == 0)
  {
    nSweeps = matX.n_rows - 1;
  }

  // Run RADICAL.
  Radical rad(noiseStdDev, nReplicates, nAngles, nSweeps);
  mat matY;
  mat matW;
  rad.DoRadical(matX, matY, matW);

  // Save results.
  if (IO::HasParam("output_ic"))
    IO::GetParam<mat>("output_ic") = std::move(matY);

  if (IO::HasParam("output_unmixing"))
    IO::GetParam<mat>("output_unmixing") = std::move(matW);

  if (IO::HasParam("objective"))
  {
    // Compute and print objective.
    mat matYT = trans(matY);
    double valEst = 0;
    for (size_t i = 0; i < matYT.n_cols; ++i)
    {
      vec y = vec(matYT.col(i));
      valEst += rad.Vasicek(y);
    }

    // Force output even if --verbose is not given.
    const bool ignoring = Log::Info.ignoreInput;
    Log::Info.ignoreInput = false;
    Log::Info << "Objective (estimate): " << valEst << "." << endl;
    Log::Info.ignoreInput = ignoring;
  }
}
