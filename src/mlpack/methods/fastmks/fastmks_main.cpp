/**
 * @file methods/fastmks/fastmks_main.cpp
 * @author Ryan Curtin
 *
 * Main executable for maximum inner product search.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "fastmks.hpp"
#include "fastmks_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::fastmks;
using namespace mlpack::kernel;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;

// Program Name.
BINDING_NAME("FastMKS (Fast Max-Kernel Search)");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the single-tree and dual-tree fast max-kernel search"
    " (FastMKS) algorithm.  Given a set of reference points and a set of query"
    " points, this can find the reference point with maximum kernel value for "
    "each query point; trained models can be reused for future queries.");

// Long description.
BINDING_LONG_DESC(
    "This program will find the k maximum kernels of a set of points, "
    "using a query set and a reference set (which can optionally be the same "
    "set). More specifically, for each point in the query set, the k points in"
    " the reference set with maximum kernel evaluations are found.  The kernel "
    "function used is specified with the " + PRINT_PARAM_STRING("kernel") +
    " parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, the following command will calculate, for each point in the "
    "query set " + PRINT_DATASET("query") + ", the five points in the "
    "reference set " + PRINT_DATASET("reference") + " with maximum kernel "
    "evaluation using the linear kernel.  The kernel evaluations may be saved "
    "with the  " + PRINT_DATASET("kernels") + " output parameter and the "
    "indices may be saved with the " + PRINT_DATASET("indices") + " output "
    "parameter."
    "\n\n" +
    PRINT_CALL("fastmks", "k", 5, "reference", "reference", "query", "query",
        "indices", "indices", "kernels", "kernels", "kernel", "linear") +
    "\n\n"
    "The output matrices are organized such that row i and column j in the "
    "indices matrix corresponds to the index of the point in the reference set "
    "that has j'th largest kernel evaluation with the point in the query set "
    "with index i.  Row i and column j in the kernels matrix corresponds to the"
    " kernel evaluation between those two points."
    "\n\n"
    "This program performs FastMKS using a cover tree.  The base used to build "
    "the cover tree can be specified with the " + PRINT_PARAM_STRING("base") +
    " parameter.");

// See also...
BINDING_SEE_ALSO("Fast max-kernel search tutorial (fastmks)",
        "@doxygen/fmkstutorial.html");
BINDING_SEE_ALSO("k-nearest-neighbor search", "#knn");
BINDING_SEE_ALSO("Dual-tree Fast Exact Max-Kernel Search (pdf)",
        "http://mlpack.org/papers/fmks.pdf");
BINDING_SEE_ALSO("mlpack::fastmks::FastMKS class documentation",
        "@doxygen/classmlpack_1_1fastmks_1_1FastMKS.html");

// Model-building parameters.
PARAM_MATRIX_IN("reference", "The reference dataset.", "r");
PARAM_STRING_IN("kernel", "Kernel type to use: 'linear', 'polynomial', "
    "'cosine', 'gaussian', 'epanechnikov', 'triangular', 'hyptan'.", "K",
    "linear");
PARAM_DOUBLE_IN("base", "Base to use during cover tree construction.", "b",
    2.0);

// Kernel parameters.
PARAM_DOUBLE_IN("degree", "Degree of polynomial kernel.", "d", 2.0);
PARAM_DOUBLE_IN("offset", "Offset of kernel (for polynomial and hyptan "
    "kernels).", "o", 0.0);
PARAM_DOUBLE_IN("bandwidth", "Bandwidth (for Gaussian, Epanechnikov, and "
    "triangular kernels).", "w", 1.0);
PARAM_DOUBLE_IN("scale", "Scale of kernel (for hyptan kernel).", "s", 1.0);

// Load/save models.
PARAM_MODEL_IN(FastMKSModel, "input_model", "Input FastMKS model to use.", "m");
PARAM_MODEL_OUT(FastMKSModel, "output_model", "Output for FastMKS model.", "M");

// Search preferences.
PARAM_MATRIX_IN("query", "The query dataset.", "q");
PARAM_INT_IN("k", "Number of maximum kernels to find.", "k", 0);
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single", "If true, single-tree search is used (as opposed to "
    "dual-tree search.", "S");

PARAM_MATRIX_OUT("kernels", "Output matrix of kernels.", "p");
PARAM_UMATRIX_OUT("indices", "Output matrix of indices.", "i");

static void mlpackMain()
{
  // Validate command-line parameters.
  RequireOnlyOnePassed({ "reference", "input_model" }, true);

  ReportIgnoredParam({{ "input_model", true }}, "kernel");
  ReportIgnoredParam({{ "input_model", true }}, "bandwidth");
  ReportIgnoredParam({{ "input_model", true }}, "degree");
  ReportIgnoredParam({{ "input_model", true }}, "offset");

  ReportIgnoredParam({{ "k", false }}, "indices");
  ReportIgnoredParam({{ "k", false }}, "kernels");
  ReportIgnoredParam({{ "k", false }}, "query");

  if (IO::HasParam("k"))
  {
    RequireAtLeastOnePassed({ "indices", "kernels" }, false,
        "no output will be saved");
  }

  // Check on kernel type.
  RequireParamInSet<string>("kernel", { "linear", "polynomial", "cosine",
      "gaussian", "triangular", "hyptan", "epanechnikov" }, true,
      "unknown kernel type");

  // Make sure number of maximum kernels is greater than 0.
  if (IO::HasParam("k"))
  {
    RequireParamValue<int>("k", [](int x) { return x > 0; }, true,
        "number of maximum kernels must be greater than 0");
  }

  if (IO::HasParam("base"))
  {
    RequireParamValue<double>("base", [](double x) { return x > 1.0; }, true,
        "base must be greater than or equal to 1!");
  }

  // Naive mode overrides single mode.
  ReportIgnoredParam({{ "naive", true }}, "single");

  FastMKSModel* model;
  arma::mat referenceData;
  if (IO::HasParam("reference"))
  {
    model = new FastMKSModel();
    referenceData = std::move(IO::GetParam<arma::mat>("reference"));

    Log::Info << "Loaded reference data (" << referenceData.n_rows << " x "
        << referenceData.n_cols << ")." << endl;

    // For cover tree construction.
    const double base = IO::GetParam<double>("base");

    // Kernel parameters.
    const string kernelType = IO::GetParam<string>("kernel");
    const double degree = IO::GetParam<double>("degree");
    const double offset = IO::GetParam<double>("offset");
    const double bandwidth = IO::GetParam<double>("bandwidth");
    const double scale = IO::GetParam<double>("scale");

    // Search preferences.
    const bool naive = IO::HasParam("naive");
    const bool single = IO::HasParam("single");

    if (kernelType == "linear")
    {
      LinearKernel lk;
      model->KernelType() = FastMKSModel::LINEAR_KERNEL;
      model->BuildModel(std::move(referenceData), lk, single, naive, base);
    }
    else if (kernelType == "polynomial")
    {
      PolynomialKernel pk(degree, offset);
      model->KernelType() = FastMKSModel::POLYNOMIAL_KERNEL;
      model->BuildModel(std::move(referenceData), pk, single, naive, base);
    }
    else if (kernelType == "cosine")
    {
      CosineDistance cd;
      model->KernelType() = FastMKSModel::COSINE_DISTANCE;
      model->BuildModel(std::move(referenceData), cd, single, naive, base);
    }
    else if (kernelType == "gaussian")
    {
      GaussianKernel gk(bandwidth);
      model->KernelType() = FastMKSModel::GAUSSIAN_KERNEL;
      model->BuildModel(std::move(referenceData), gk, single, naive, base);
    }
    else if (kernelType == "epanechnikov")
    {
      EpanechnikovKernel ek(bandwidth);
      model->KernelType() = FastMKSModel::EPANECHNIKOV_KERNEL;
      model->BuildModel(std::move(referenceData), ek, single, naive, base);
    }
    else if (kernelType == "triangular")
    {
      TriangularKernel tk(bandwidth);
      model->KernelType() = FastMKSModel::TRIANGULAR_KERNEL;
      model->BuildModel(std::move(referenceData), tk, single, naive, base);
    }
    else if (kernelType == "hyptan")
    {
      HyperbolicTangentKernel htk(scale, offset);
      model->KernelType() = FastMKSModel::HYPTAN_KERNEL;
      model->BuildModel(std::move(referenceData), htk, single, naive, base);
    }
  }
  else
  {
    // Load model from file, then do whatever is necessary.
    model = IO::GetParam<FastMKSModel*>("input_model");
  }

  // Set search preferences.
  model->Naive() = IO::HasParam("naive");
  model->SingleMode() = IO::HasParam("single");

  // Should we do search?
  if (IO::HasParam("k"))
  {
    arma::mat kernels;
    arma::Mat<size_t> indices;

    if (IO::HasParam("query"))
    {
      const double base = IO::GetParam<double>("base");

      arma::mat queryData = std::move(IO::GetParam<arma::mat>("query"));

      Log::Info << "Loaded query data (" << queryData.n_rows << " x "
          << queryData.n_cols << ")." << endl;

      try
      {
        model->Search(queryData, (size_t) IO::GetParam<int>("k"), indices,
            kernels, base);
      }
      catch (std::invalid_argument& e)
      {
        // Delete the memory, if needed.
        if (IO::HasParam("reference"))
          delete model;
        throw;
      }
    }
    else
    {
      try
      {
        model->Search((size_t) IO::GetParam<int>("k"), indices, kernels);
      }
      catch (std::invalid_argument& e)
      {
        // Delete the memory, if needed.
        if (IO::HasParam("reference"))
          delete model;
        throw e;
      }
    }

    // Save output.
    IO::GetParam<arma::mat>("kernels") = std::move(kernels);
    IO::GetParam<arma::Mat<size_t>>("indices") = std::move(indices);
  }

  // Save the model.
  IO::GetParam<FastMKSModel*>("output_model") = model;
}
