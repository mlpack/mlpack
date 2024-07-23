/**
 * @file methods/kernel_pca/kernel_pca_main.cpp
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

#undef BINDING_NAME
#define BINDING_NAME kernel_pca

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/nystroem_method/nystroem_method.hpp>
#include <mlpack/methods/kernel_pca/kernel_rules/nystroem_method.hpp>

#include "kernel_pca.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Kernel Principal Components Analysis");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of Kernel Principal Components Analysis (KPCA).  This "
    "can be used to perform nonlinear dimensionality reduction or preprocessing"
    " on a given dataset.");

// Long description.
BINDING_LONG_DESC(
    "This program performs Kernel Principal Components Analysis (KPCA) on the "
    "specified dataset with the specified kernel.  This will transform the "
    "data onto the kernel principal components, and optionally reduce the "
    "dimensionality by ignoring the kernel principal components with the "
    "smallest eigenvalues."
    "\n\n"
    "For the case where a linear kernel is used, this reduces to regular "
    "PCA."
    "\n\n"
    "The kernels that are supported are listed below:"
    "\n\n"
    " * 'linear': the standard linear dot product (same as normal PCA):\n"
    "    `K(x, y) = x^T y`\n"
    "\n"
    " * 'gaussian': a Gaussian kernel; requires bandwidth:\n"
    "    `K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))`\n"
    "\n"
    " * 'polynomial': polynomial kernel; requires offset and degree:\n"
    "    `K(x, y) = (x^T y + offset) ^ degree`\n"
    "\n"
    " * 'hyptan': hyperbolic tangent kernel; requires scale and offset:\n"
    "    `K(x, y) = tanh(scale * (x^T y) + offset)`\n"
    "\n"
    " * 'laplacian': Laplacian kernel; requires bandwidth:\n"
    "    `K(x, y) = exp(-(|| x - y ||) / bandwidth)`\n"
    "\n"
    " * 'epanechnikov': Epanechnikov kernel; requires bandwidth:\n"
    "    `K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)`\n"
    "\n"
    " * 'cosine': cosine distance:\n"
    "    `K(x, y) = 1 - (x^T y) / (|| x || * || y ||)`\n"
    "\n"
    "The parameters for each of the kernels should be specified with the "
    "options " + PRINT_PARAM_STRING("bandwidth") + ", " +
    PRINT_PARAM_STRING("kernel_scale") + ", " +
    PRINT_PARAM_STRING("offset") + ", or " + PRINT_PARAM_STRING("degree") +
    " (or a combination of those parameters)."
    "\n\n"
    "Optionally, the Nystroem method (\"Using the Nystroem method to speed "
    "up kernel machines\", 2001) can be used to calculate the kernel matrix by "
    "specifying the " + PRINT_PARAM_STRING("nystroem_method") + " parameter. "
    "This approach works by using a subset of the data as basis to reconstruct "
    "the kernel matrix; to specify the sampling scheme, the " +
    PRINT_PARAM_STRING("sampling") + " parameter is used.  The "
    "sampling scheme for the Nystroem method can be chosen from the "
    "following list: 'kmeans', 'random', 'ordered'.");

// Example.
BINDING_EXAMPLE(
    "For example, the following command will perform KPCA on the dataset " +
    PRINT_DATASET("input") + " using the Gaussian kernel, and saving the "
    "transformed data to " + PRINT_DATASET("transformed") + ": "
    "\n\n" +
    PRINT_CALL("kernel_pca", "input", "input", "kernel", "gaussian", "output",
        "transformed"));

// See also...
BINDING_SEE_ALSO("Kernel principal component analysis on Wikipedia",
    "https://en.wikipedia.org/wiki/Kernel_principal_component_analysis");
BINDING_SEE_ALSO("Nonlinear Component Analysis as a Kernel Eigenvalue "
    "Problem", "https://www.mlpack.org/papers/kpca.pdf");
BINDING_SEE_ALSO("KernelPCA class documentation",
    "@src/mlpack/methods/kernel_pca/kernel_pca.hpp");

PARAM_MATRIX_IN_REQ("input", "Input dataset to perform KPCA on.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save modified dataset to.", "o");
PARAM_STRING_IN_REQ("kernel", "The kernel to use; see the above documentation "
    "for the list of usable kernels.", "k");

PARAM_INT_IN("new_dimensionality", "If not 0, reduce the dimensionality of "
    "the output dataset by ignoring the dimensions with the smallest "
    "eigenvalues.", "d", 0);

PARAM_FLAG("center", "If set, the transformed data will be centered about the "
    "origin.", "c");

PARAM_FLAG("nystroem_method", "If set, the Nystroem method will be used.", "n");

PARAM_STRING_IN("sampling", "Sampling scheme to use for the Nystroem method: "
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
          KMeansSelection<> > > kpca(kernel, centerTransformedData);
      kpca.Apply(dataset, newDim);
    }
    else if (sampling == "random")
    {
      KernelPCA<KernelType, NystroemKernelRule<KernelType,
          RandomSelection> > kpca(kernel, centerTransformedData);
      kpca.Apply(dataset, newDim);
    }
    else if (sampling == "ordered")
    {
      KernelPCA<KernelType, NystroemKernelRule<KernelType,
          OrderedSelection> > kpca(kernel, centerTransformedData);
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

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  // Load input dataset.
  mat dataset = std::move(params.Get<arma::mat>("input"));

  // Get the new dimensionality, if it is necessary.
  size_t newDim = dataset.n_rows;
  if (params.Get<int>("new_dimensionality") != 0)
  {
    newDim = params.Get<int>("new_dimensionality");

    if (newDim > dataset.n_rows)
    {
      Log::Fatal << "New dimensionality (" << newDim
          << ") cannot be greater than existing dimensionality ("
          << dataset.n_rows << ")!" << endl;
    }
  }

  // Get the kernel type and make sure it is valid.
  RequireParamInSet<string>(params, "kernel", { "linear", "gaussian",
      "polynomial", "hyptan", "laplacian", "epanechnikov", "cosine" }, true,
      "unknown kernel type");
  const string kernelType = params.Get<string>("kernel");

  const bool centerTransformedData = params.Has("center");
  const bool nystroem = params.Has("nystroem_method");
  const string sampling = params.Get<string>("sampling");

  if (kernelType == "linear")
  {
    LinearKernel kernel;
    RunKPCA<LinearKernel>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else if (kernelType == "gaussian")
  {
    const double bandwidth = params.Get<double>("bandwidth");

    GaussianKernel kernel(bandwidth);
    RunKPCA<GaussianKernel>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else if (kernelType == "polynomial")
  {
    const double degree = params.Get<double>("degree");
    const double offset = params.Get<double>("offset");

    PolynomialKernel kernel(degree, offset);
    RunKPCA<PolynomialKernel>(dataset, centerTransformedData, nystroem,
        newDim, sampling, kernel);
  }
  else if (kernelType == "hyptan")
  {
    const double scale = params.Get<double>("kernel_scale");
    const double offset = params.Get<double>("offset");

    HyperbolicTangentKernel kernel(scale, offset);
    RunKPCA<HyperbolicTangentKernel>(dataset, centerTransformedData, nystroem,
        newDim, sampling, kernel);
  }
  else if (kernelType == "laplacian")
  {
    const double bandwidth = params.Get<double>("bandwidth");

    LaplacianKernel kernel(bandwidth);
    RunKPCA<LaplacianKernel>(dataset, centerTransformedData, nystroem, newDim,
        sampling, kernel);
  }
  else if (kernelType == "epanechnikov")
  {
    const double bandwidth = params.Get<double>("bandwidth");

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

  // Save the output dataset.
  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(dataset);
}
