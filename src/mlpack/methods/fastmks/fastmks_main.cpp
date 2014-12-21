/**
 * @file fastmks_main.cpp
 * @author Ryan Curtin
 *
 * Main executable for maximum inner product search.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include "fastmks.hpp"

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
    "  --indices_file indices.csv --products_file kernels.csv --kernel linear"
    "\n\n"
    "The output files are organized such that row i and column j in the indices"
    " output file corresponds to the index of the point in the reference set "
    "that has i'th largest kernel evaluation with the point in the query set "
    "with index j.  Row i and column j in the products output file corresponds "
    "to the kernel evaluation between those two points."
    "\n\n"
    "This executable performs FastMKS using a cover tree.  The base used to "
    "build the cover tree can be specified with the --base option.");

// Define our input parameters.
PARAM_STRING_REQ("reference_file", "File containing the reference dataset.",
    "r");
PARAM_STRING("query_file", "File containing the query dataset.", "q", "");

PARAM_INT_REQ("k", "Number of maximum inner products to find.", "k");

PARAM_STRING("products_file", "File to save inner products into.", "p", "");
PARAM_STRING("indices_file", "File to save indices of inner products into.",
    "i", "");

PARAM_STRING("kernel", "Kernel type to use: 'linear', 'polynomial', 'cosine', "
    "'gaussian', 'epanechnikov', 'triangular', 'hyptan'.", "K", "linear");

PARAM_FLAG("naive", "If true, O(n^2) naive mode is used for computation.", "N");
PARAM_FLAG("single", "If true, single-tree search is used (as opposed to "
    "dual-tree search.", "S");

// Cover tree parameter.
PARAM_DOUBLE("base", "Base to use during cover tree construction.", "b", 2.0);

// Kernel parameters.
PARAM_DOUBLE("degree", "Degree of polynomial kernel.", "d", 2.0);
PARAM_DOUBLE("offset", "Offset of kernel (for polynomial and hyptan kernels).",
    "o", 0.0);
PARAM_DOUBLE("bandwidth", "Bandwidth (for Gaussian, Epanechnikov, and "
    "triangular kernels).", "w", 1.0);
PARAM_DOUBLE("scale", "Scale of kernel (for hyptan kernel).", "s", 1.0);

//! Run FastMKS on a single dataset for the given kernel type.
template<typename KernelType>
void RunFastMKS(const arma::mat& referenceData,
                const bool single,
                const bool naive,
                const double base,
                const size_t k,
                arma::Mat<size_t>& indices,
                arma::mat& products,
                KernelType& kernel)
{
  // Create the tree with the specified base.
  typedef CoverTree<IPMetric<KernelType>, FirstPointIsRoot, FastMKSStat>
      TreeType;
  IPMetric<KernelType> metric(kernel);
  TreeType tree(referenceData, metric, base);

  // Create FastMKS object.
  FastMKS<KernelType> fastmks(referenceData, &tree, (single && !naive), naive);

  // Now search with it.
  fastmks.Search(k, indices, products);
}

//! Run FastMKS for a given query and reference set using the given kernel type.
template<typename KernelType>
void RunFastMKS(const arma::mat& referenceData,
                const arma::mat& queryData,
                const bool single,
                const bool naive,
                const double base,
                const size_t k,
                arma::Mat<size_t>& indices,
                arma::mat& products,
                KernelType& kernel)
{
  // Create the tree with the specified base.
  typedef CoverTree<IPMetric<KernelType>, FirstPointIsRoot, FastMKSStat>
      TreeType;
  IPMetric<KernelType> metric(kernel);
  TreeType referenceTree(referenceData, metric, base);
  TreeType queryTree(queryData, metric, base);

  // Create FastMKS object.
  FastMKS<KernelType> fastmks(referenceData, &referenceTree, queryData,
      &queryTree, (single && !naive), naive);

  // Now search with it.
  fastmks.Search(k, indices, products);
}

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  // Get reference dataset filename.
  const string referenceFile = CLI::GetParam<string>("reference_file");

  // The number of max kernel values to find.
  const size_t k = CLI::GetParam<int>("k");

  // Runtime parameters.
  const bool naive = CLI::HasParam("naive");
  const bool single = CLI::HasParam("single");

  // For cover tree construction.
  const double base = CLI::GetParam<double>("base");

  // Kernel parameters.
  const string kernelType = CLI::GetParam<string>("kernel");
  const double degree = CLI::GetParam<double>("degree");
  const double offset = CLI::GetParam<double>("offset");
  const double bandwidth = CLI::GetParam<double>("bandwidth");
  const double scale = CLI::GetParam<double>("scale");

  // The datasets.  The query matrix may never be used.
  arma::mat referenceData;
  arma::mat queryData;

  data::Load(referenceFile, referenceData, true);

  Log::Info << "Loaded reference data from '" << referenceFile << "' ("
      << referenceData.n_rows << " x " << referenceData.n_cols << ")." << endl;

  // Sanity check on k value.
  if (k > referenceData.n_cols)
  {
    Log::Fatal << "Invalid k: " << k << "; must be greater than 0 and less ";
    Log::Fatal << "than or equal to the number of reference points (";
    Log::Fatal << referenceData.n_cols << ")." << endl;
  }

  // Check on kernel type.
  if ((kernelType != "linear") && (kernelType != "polynomial") &&
      (kernelType != "cosine") && (kernelType != "gaussian") &&
      (kernelType != "graph") && (kernelType != "approxGraph") &&
      (kernelType != "triangular") && (kernelType != "hyptan") &&
      (kernelType != "inv-mq") && (kernelType != "epanechnikov"))
  {
    Log::Fatal << "Invalid kernel type: '" << kernelType << "'; must be ";
    Log::Fatal << "'linear' or 'polynomial'." << endl;
  }

  // Load the query matrix, if we can.
  if (CLI::HasParam("query_file"))
  {
    const string queryFile = CLI::GetParam<string>("query_file");
    data::Load(queryFile, queryData, true);

    Log::Info << "Loaded query data from '" << queryFile << "' ("
        << queryData.n_rows << " x " << queryData.n_cols << ")." << endl;
  }
  else
  {
    Log::Info << "Using reference dataset as query dataset (--query_file not "
        << "specified)." << endl;
  }

  // Naive mode overrides single mode.
  if (naive && single)
  {
    Log::Warn << "--single ignored because --naive is present." << endl;
  }

  // Matrices for output storage.
  arma::Mat<size_t> indices;
  arma::mat products;

  // Construct FastMKS object.
  if (queryData.n_elem == 0)
  {
    if (kernelType == "linear")
    {
      LinearKernel lk;
      RunFastMKS<LinearKernel>(referenceData, single, naive, base, k, indices,
          products, lk);
    }
    else if (kernelType == "polynomial")
    {

      PolynomialKernel pk(degree, offset);
      RunFastMKS<PolynomialKernel>(referenceData, single, naive, base, k,
          indices, products, pk);
    }
    else if (kernelType == "cosine")
    {
      CosineDistance cd;
      RunFastMKS<CosineDistance>(referenceData, single, naive, base, k, indices,
          products, cd);
    }
    else if (kernelType == "gaussian")
    {
      GaussianKernel gk(bandwidth);
      RunFastMKS<GaussianKernel>(referenceData, single, naive, base, k, indices,
          products, gk);
    }
    else if (kernelType == "epanechnikov")
    {
      EpanechnikovKernel ek(bandwidth);
      RunFastMKS<EpanechnikovKernel>(referenceData, single, naive, base, k,
          indices, products, ek);
    }
    else if (kernelType == "triangular")
    {
      TriangularKernel tk(bandwidth);
      RunFastMKS<TriangularKernel>(referenceData, single, naive, base, k,
          indices, products, tk);
    }
    else if (kernelType == "hyptan")
    {
      HyperbolicTangentKernel htk(scale, offset);
      RunFastMKS<HyperbolicTangentKernel>(referenceData, single, naive, base, k,
          indices, products, htk);
    }
  }
  else
  {
    if (kernelType == "linear")
    {
      LinearKernel lk;
      RunFastMKS<LinearKernel>(referenceData, queryData, single, naive, base, k,
          indices, products, lk);
    }
    else if (kernelType == "polynomial")
    {
      PolynomialKernel pk(degree, offset);
      RunFastMKS<PolynomialKernel>(referenceData, queryData, single, naive,
          base, k, indices, products, pk);
    }
    else if (kernelType == "cosine")
    {
      CosineDistance cd;
      RunFastMKS<CosineDistance>(referenceData, queryData, single, naive, base,
          k, indices, products, cd);
    }
    else if (kernelType == "gaussian")
    {
      GaussianKernel gk(bandwidth);
      RunFastMKS<GaussianKernel>(referenceData, queryData, single, naive, base,
          k, indices, products, gk);
    }
    else if (kernelType == "epanechnikov")
    {
      EpanechnikovKernel ek(bandwidth);
      RunFastMKS<EpanechnikovKernel>(referenceData, queryData, single, naive,
          base, k, indices, products, ek);
    }
    else if (kernelType == "triangular")
    {
      TriangularKernel tk(bandwidth);
      RunFastMKS<TriangularKernel>(referenceData, queryData, single, naive,
          base, k, indices, products, tk);
    }
    else if (kernelType == "hyptan")
    {
      HyperbolicTangentKernel htk(scale, offset);
      RunFastMKS<HyperbolicTangentKernel>(referenceData, queryData, single,
          naive, base, k, indices, products, htk);
    }
  }

  // Save output, if we were asked to.
  if (CLI::HasParam("products_file"))
  {
    const string productsFile = CLI::GetParam<string>("products_file");
    data::Save(productsFile, products, false);
  }

  if (CLI::HasParam("indices_file"))
  {
    const string indicesFile = CLI::GetParam<string>("indices_file");
    data::Save(indicesFile, indices, false);
  }
}
