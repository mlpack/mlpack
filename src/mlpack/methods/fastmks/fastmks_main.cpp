/**
 * @file fastmks_main.cpp
 * @author Ryan Curtin
 *
 * Main executable for maximum inner product search.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "fastmks.hpp"
#include "fastmks_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::fastmks;
using namespace mlpack::kernel;
using namespace mlpack::tree;
using namespace mlpack::metric;

PROGRAM_INFO("FastMKS (Fast Max-Kernel Search)",
    "This program will find the k maximum kernel of a set of points, "
    "using a query set and a reference set (which can optionally be the same "
    "set). More specifically, for each point in the query set, the k points in"
    " the reference set with maximum kernel evaluations are found.  The kernel "
    "function used is specified by --kernel."
    "\n\n"
    "For example, the following command will calculate, for each point in "
    "'query.csv', the five points in 'reference.csv' with maximum kernel "
    "evaluation using the linear kernel.  The kernel evaluations are stored in "
    "'kernels.csv' and the indices are stored in 'indices.csv'."
    "\n\n"
    "$ fastmks --k 5 --reference_file reference.csv --query_file query.csv\n"
    "  --indices_file indices.csv --kernels_file kernels.csv --kernel linear"
    "\n\n"
    "The output files are organized such that row i and column j in the indices"
    " output file corresponds to the index of the point in the reference set "
    "that has i'th largest kernel evaluation with the point in the query set "
    "with index j.  Row i and column j in the kernels output file corresponds "
    "to the kernel evaluation between those two points."
    "\n\n"
    "This executable performs FastMKS using a cover tree.  The base used to "
    "build the cover tree can be specified with the --base option.");

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
PARAM_STRING_IN("input_model_file", "File containing FastMKS model.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save FastMKS model to.", "M");

// Search preferences.
PARAM_MATRIX_IN("query", "The query dataset.", "q");
PARAM_INT_IN("k", "Number of maximum kernels to find.", "k", 0);
PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single", "If true, single-tree search is used (as opposed to "
    "dual-tree search.", "S");

PARAM_MATRIX_OUT("kernels", "Output matrix of kernels.", "p");
PARAM_UMATRIX_OUT("indices", "Output matrix of indices.", "i");

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Validate command-line parameters.
  if (CLI::HasParam("reference") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Cannot specify both --reference_file (-r) and "
        << "--input_model_file (-m)!" << endl;

  if (!CLI::HasParam("reference") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Must specify either --reference_file (-r) or "
        << "--input_model_file (-m)!" << endl;

  if (CLI::HasParam("input_model_file"))
  {
    if (CLI::HasParam("kernel"))
      Log::Warn << "--kernel (-k) ignored because --input_model_file (-m) is "
          << "specified." << endl;
    if (CLI::HasParam("bandwidth"))
      Log::Warn << "--bandwidth (-w) ignored because --input_model_file (-m) is"
          << " specified." << endl;
    if (CLI::HasParam("degree"))
      Log::Warn << "--degree (-d) ignored because --input_model_file (-m) is "
          << " specified." << endl;
    if (CLI::HasParam("offset"))
      Log::Warn << "--offset (-o) ignored because --input_model_file (-m) is "
          << " specified." << endl;
  }

  if (!CLI::HasParam("k") &&
      (CLI::HasParam("indices") || CLI::HasParam("kernels")))
    Log::Warn << "--indices_file and --kernels_file ignored, because no search "
        << "task is specified (i.e., --k is not specified)!" << endl;

  if (CLI::HasParam("k") &&
      !(CLI::HasParam("indices") || CLI::HasParam("kernels")))
    Log::Warn << "Search specified with --k, but no output will be saved "
        << "because neither --indices_file nor --kernels_file are specified!"
        << endl;

  if (CLI::HasParam("query") && !CLI::HasParam("k"))
    Log::Warn << "--query_file ignored, because no search task is specified "
        << "(i.e., --k is not specified)!" << endl;

  // Check on kernel type.
  const string kernelType = CLI::GetParam<string>("kernel");
  if ((kernelType != "linear") && (kernelType != "polynomial") &&
      (kernelType != "cosine") && (kernelType != "gaussian") &&
      (kernelType != "triangular") && (kernelType != "hyptan") &&
      (kernelType != "epanechnikov"))
  {
    Log::Fatal << "Invalid kernel type: '" << kernelType << "'; must be "
        << "'linear', 'polynomial', 'cosine', 'gaussian', 'triangular', or "
        << "'epanechnikov'." << endl;
  }

  // Naive mode overrides single mode.
  if (CLI::HasParam("naive") && CLI::HasParam("single"))
    Log::Warn << "--single ignored because --naive is present." << endl;

  FastMKSModel model;
  arma::mat referenceData;
  if (CLI::HasParam("reference"))
  {
    referenceData = std::move(CLI::GetParam<arma::mat>("reference"));

    Log::Info << "Loaded reference data (" << referenceData.n_rows << " x "
        << referenceData.n_cols << ")." << endl;

    // For cover tree construction.
    const double base = CLI::GetParam<double>("base");

    // Kernel parameters.
    const string kernelType = CLI::GetParam<string>("kernel");
    const double degree = CLI::GetParam<double>("degree");
    const double offset = CLI::GetParam<double>("offset");
    const double bandwidth = CLI::GetParam<double>("bandwidth");
    const double scale = CLI::GetParam<double>("scale");

    // Search preferences.
    const bool naive = CLI::HasParam("naive");
    const bool single = CLI::HasParam("single");

    if (kernelType == "linear")
    {
      LinearKernel lk;
      model.KernelType() = FastMKSModel::LINEAR_KERNEL;
      model.BuildModel(referenceData, lk, single, naive, base);
    }
    else if (kernelType == "polynomial")
    {
      PolynomialKernel pk(degree, offset);
      model.KernelType() = FastMKSModel::POLYNOMIAL_KERNEL;
      model.BuildModel(referenceData, pk, single, naive, base);
    }
    else if (kernelType == "cosine")
    {
      CosineDistance cd;
      model.KernelType() = FastMKSModel::COSINE_DISTANCE;
      model.BuildModel(referenceData, cd, single, naive, base);
    }
    else if (kernelType == "gaussian")
    {
      GaussianKernel gk(bandwidth);
      model.KernelType() = FastMKSModel::GAUSSIAN_KERNEL;
      model.BuildModel(referenceData, gk, single, naive, base);
    }
    else if (kernelType == "epanechnikov")
    {
      EpanechnikovKernel ek(bandwidth);
      model.KernelType() = FastMKSModel::EPANECHNIKOV_KERNEL;
      model.BuildModel(referenceData, ek, single, naive, base);
    }
    else if (kernelType == "triangular")
    {
      TriangularKernel tk(bandwidth);
      model.KernelType() = FastMKSModel::TRIANGULAR_KERNEL;
      model.BuildModel(referenceData, tk, single, naive, base);
    }
    else if (kernelType == "hyptan")
    {
      HyperbolicTangentKernel htk(scale, offset);
      model.KernelType() = FastMKSModel::HYPTAN_KERNEL;
      model.BuildModel(referenceData, htk, single, naive, base);
    }
  }
  else
  {
    // Load model from file, then do whatever is necessary.
    data::Load(CLI::GetParam<string>("input_model_file"), "fastmks_model",
        model, true);
  }

  // Set search preferences.
  model.Naive() = CLI::HasParam("naive");
  model.SingleMode() = CLI::HasParam("single");

  // Should we do search?
  if (CLI::HasParam("k"))
  {
    arma::mat kernels;
    arma::Mat<size_t> indices;

    if (CLI::HasParam("query"))
    {
      const double base = CLI::GetParam<double>("base");

      arma::mat queryData = std::move(CLI::GetParam<arma::mat>("query"));

      Log::Info << "Loaded query data (" << queryData.n_rows << " x "
          << queryData.n_cols << ")." << endl;

      model.Search(queryData, (size_t) CLI::GetParam<int>("k"), indices,
          kernels, base);
    }
    else
    {
      model.Search((size_t) CLI::GetParam<int>("k"), indices, kernels);
    }

    // Save output, if we were asked to.
    if (CLI::HasParam("kernels"))
      CLI::GetParam<arma::mat>("kernels") = std::move(kernels);

    if (CLI::HasParam("indices"))
      CLI::GetParam<arma::Mat<size_t>>("indices") = std::move(indices);
  }

  // Save the model, if requested.
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "fastmks_model",
        model);
}
