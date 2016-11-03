/**
 * @file kernel_pca_main.cpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Executable for Kernel PCA.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/nystroem_method/ordered_selection.hpp>
#include <mlpack/methods/nystroem_method/random_selection.hpp>
#include <mlpack/methods/nystroem_method/kmeans_selection.hpp>
#include <mlpack/methods/nystroem_method/nystroem_method.hpp>
#include <mlpack/methods/kernel_pca/kernel_rules/nystroem_method.hpp>

#include "kernel_pca.hpp"

using namespace mlpack;
using namespace mlpack::kpca;
using namespace mlpack::kernel;
using namespace std;
using namespace arma;

PROGRAM_INFO("Kernel Principal Components Analysis",
    "This program performs Kernel Principal Components Analysis (KPCA) on the "
    "specified dataset with the specified kernel.  This will transform the "
    "data onto the kernel principal components, and optionally reduce the "
    "dimensionality by ignoring the kernel principal components with the "
    "smallest eigenvalues."
    "\n\n"
    "For the case where a linear kernel is used, this reduces to regular "
    "PCA."
    "\n\n"
    "For example, the following will perform KPCA on the 'input.csv' file using"
    " the gaussian kernel and store the transformed date in the "
    "'transformed.csv' file."
    "\n\n"
    "$ kernel_pca -i input.csv -k gaussian -o transformed.csv"
    "\n\n"
    "The kernels that are supported are listed below:"
    "\n\n"
    " * 'linear': the standard linear dot product (same as normal PCA):\n"
    "    K(x, y) = x^T y\n"
    "\n"
    " * 'gaussian': a Gaussian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))\n"
    "\n"
    " * 'polynomial': polynomial kernel; requires offset and degree:\n"
    "    K(x, y) = (x^T y + offset) ^ degree\n"
    "\n"
    " * 'hyptan': hyperbolic tangent kernel; requires scale and offset:\n"
    "    K(x, y) = tanh(scale * (x^T y) + offset)\n"
    "\n"
    " * 'laplacian': Laplacian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y ||) / bandwidth)\n"
    "\n"
    " * 'epanechnikov': Epanechnikov kernel; requires bandwidth:\n"
    "    K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)\n"
    "\n"
    " * 'cosine': cosine distance:\n"
    "    K(x, y) = 1 - (x^T y) / (|| x || * || y ||)\n"
    "\n"
    "The parameters for each of the kernels should be specified with the "
    "options --bandwidth, --kernel_scale, --offset, or --degree (or a "
    "combination of those options)."
    "\n\n"
    "Optionally, the nystr\u00F6m method (\"Using the Nystroem method to speed up"
    " kernel machines\", 2001) can be used to calculate the kernel matrix by "
    "specifying the --nystroem_method (-n) option. This approach works by using"
    " a subset of the data as basis to reconstruct the kernel matrix; to "
    "specify the sampling scheme, the --sampling parameter is used, the "
    "sampling scheme for the nystr\u00F6m method can be chosen from the following"
    " list: kmeans, random, ordered.");

PARAM_MATRIX_IN_REQ("input", "Input dataset to perform KPCA on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_STRING_IN_REQ("kernel", "The kernel to use; see the above documentation "
    "for the list of usable kernels.", "k");

PARAM_INT_IN("new_dimensionality", "If not 0, reduce the dimensionality of "
    "the output dataset by ignoring the dimensions with the smallest "
    "eigenvalues.", "d", 0);

PARAM_FLAG("center", "If set, the transformed data will be centered about the "
    "origin.", "c");

PARAM_FLAG("nystroem_method", "If set, the nystroem method will be used.", "n");

PARAM_STRING_IN("sampling", "Sampling scheme to use for the nystroem method: "
    "'kmeans', 'random', 'ordered'", "s", "kmeans");

PARAM_DOUBLE_IN("kernel_scale", "Scale, for 'hyptan' kernel.", "S", 1.0);
PARAM_DOUBLE_IN("offset", "Offset, for 'hyptan' and 'polynomial' kernels.", "O",
    0.0);
PARAM_DOUBLE_IN("bandwidth", "Bandwidth, for 'gaussian' and 'laplacian' "
    "kernels.", "b", 1.0);
PARAM_DOUBLE_IN("degree", "Degree of polynomial, for 'polynomial' kernel.", "D",
    1.0);

//! Run RunKPCA on the specified dataset for the given kernel type.
template<typename KernelType>
void RunKPCA(arma::mat& dataset,
             const bool centerTransformedData,
             const bool nystroem,
             const size_t newDim,
             const string& sampling,
             KernelType& kernel)
{
  if (nystroem)
  {
    // Make sure the sampling scheme is valid.
    if (sampling == "kmeans")
    {
      KernelPCA<KernelType, NystroemKernelRule<KernelType,
          KMeansSelection<> > >kpca;
      kpca.Apply(dataset, newDim);
    }
    else if (sampling == "random")
    {
      KernelPCA<KernelType, NystroemKernelRule<KernelType,
          RandomSelection> > kpca;
      kpca.Apply(dataset, newDim);
    }
    else if (sampling == "ordered")
    {
      KernelPCA<KernelType, NystroemKernelRule<KernelType,
          OrderedSelection> > kpca;
      kpca.Apply(dataset, newDim);
    }
    else
    {
      // Invalid sampling scheme.
      Log::Fatal << "Invalid sampling scheme ('" << sampling << "'); valid "
        << "choices are 'kmeans', 'random' and 'ordered'" << endl;
    }
  }
  else
  {
    KernelPCA<KernelType> kpca(kernel, centerTransformedData);
    kpca.Apply(dataset, newDim);
  }
}

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  if (!CLI::HasParam("output"))
    Log::Warn << "--output_file is not specified; no output will be saved!"
        << endl;

  // Load input dataset.
  mat dataset = std::move(CLI::GetParam<arma::mat>("input"));

  // Get the new dimensionality, if it is necessary.
  size_t newDim = dataset.n_rows;
  if (CLI::GetParam<int>("new_dimensionality") != 0)
  {
    newDim = CLI::GetParam<int>("new_dimensionality");

    if (newDim > dataset.n_rows)
    {
      Log::Fatal << "New dimensionality (" << newDim
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!" << endl;
    }
  }

  // Get the kernel type and make sure it is valid.
  const string kernelType = CLI::GetParam<string>("kernel");

  const bool centerTransformedData = CLI::HasParam("center");
  const bool nystroem = CLI::HasParam("nystroem_method");
  const string sampling = CLI::GetParam<string>("sampling");

  if (kernelType == "linear")
  {
    LinearKernel kernel;
    RunKPCA<LinearKernel>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else if (kernelType == "gaussian")
  {
    const double bandwidth = CLI::GetParam<double>("bandwidth");

    GaussianKernel kernel(bandwidth);
    RunKPCA<GaussianKernel>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else if (kernelType == "polynomial")
  {
    const double degree = CLI::GetParam<double>("degree");
    const double offset = CLI::GetParam<double>("offset");

    PolynomialKernel kernel(degree, offset);
    RunKPCA<PolynomialKernel>(dataset, centerTransformedData, nystroem,
        newDim, sampling, kernel);
  }
  else if (kernelType == "hyptan")
  {
    const double scale = CLI::GetParam<double>("kernel_scale");
    const double offset = CLI::GetParam<double>("offset");

    HyperbolicTangentKernel kernel(scale, offset);
    RunKPCA<HyperbolicTangentKernel>(dataset, centerTransformedData, nystroem,
        newDim, sampling, kernel);
  }
  else if (kernelType == "laplacian")
  {
    const double bandwidth = CLI::GetParam<double>("bandwidth");

    LaplacianKernel kernel(bandwidth);
    RunKPCA<LaplacianKernel>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else if (kernelType == "epanechnikov")
  {
    const double bandwidth = CLI::GetParam<double>("bandwidth");

    EpanechnikovKernel kernel(bandwidth);
    RunKPCA<EpanechnikovKernel>(dataset, centerTransformedData, nystroem,
        newDim, sampling, kernel);
  }
  else if (kernelType == "cosine")
  {
    CosineDistance kernel;
    RunKPCA<CosineDistance>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else
  {
    // Invalid kernel type.
    Log::Fatal << "Invalid kernel type ('" << kernelType << "'); valid choices "
        << "are 'linear', 'gaussian', 'polynomial', 'hyptan', 'laplacian', and "
        << "'cosine'." << endl;
  }

  // Save the output dataset.
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(dataset);
}
