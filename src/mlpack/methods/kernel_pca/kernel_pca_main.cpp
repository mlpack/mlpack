/**
 * @file kernel_pca_main.cpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Executable for Kernel PCA.
 *
 * This file is part of MLPACK 1.0.4.
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
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/cosine_distance.hpp>

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
    " * 'cosine': cosine distance:\n"
    "    K(x, y) = 1 - (x^T y) / (|| x || * || y ||)\n"
    "\n"
    "The parameters for each of the kernels should be specified with the "
    "options --bandwidth, --kernel_scale, --offset, or --degree (or a "
    "combination of those options).\n");

PARAM_STRING_REQ("input_file", "Input dataset to perform KPCA on.", "i");
PARAM_STRING_REQ("output_file", "File to save modified dataset to.", "o");
PARAM_STRING_REQ("kernel", "The kernel to use; see the above documentation for "
    "the list of usable kernels.", "k");

PARAM_INT("new_dimensionality", "If not 0, reduce the dimensionality of "
    "the output dataset by ignoring the dimensions with the smallest "
    "eigenvalues.", "d", 0);

PARAM_FLAG("scale", "If set, the data will be scaled before performing KPCA "
    "such that the variance of each feature is 1.", "s");

PARAM_DOUBLE("kernel_scale", "Scale, for 'hyptan' kernel.", "S", 1.0);
PARAM_DOUBLE("offset", "Offset, for 'hyptan' and 'polynomial' kernels.", "O",
    0.0);
PARAM_DOUBLE("bandwidth", "Bandwidth, for 'gaussian' and 'laplacian' kernels.",
    "b", 1.0);
PARAM_DOUBLE("degree", "Degree of polynomial, for 'polynomial' kernel.", "d",
    1.0);

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);

  // Load input dataset.
  mat dataset;
  const string inputFile = CLI::GetParam<string>("input_file");
  data::Load(inputFile, dataset, true); // Fatal on failure.

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

  const bool scaleData = CLI::HasParam("scale");

  if (kernelType == "linear")
  {
    KernelPCA<LinearKernel> kpca(LinearKernel(), scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "gaussian")
  {
    const double bandwidth = CLI::GetParam<double>("bandwidth");

    GaussianKernel kernel(bandwidth);
    KernelPCA<GaussianKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "polynomial")
  {
    const double degree = CLI::GetParam<double>("degree");
    const double offset = CLI::GetParam<double>("offset");

    PolynomialKernel kernel(offset, degree);
    KernelPCA<PolynomialKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "hyptan")
  {
    const double scale = CLI::GetParam<double>("kernel_scale");
    const double offset = CLI::GetParam<double>("offset");

    HyperbolicTangentKernel kernel(scale, offset);
    KernelPCA<HyperbolicTangentKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "laplacian")
  {
    const double bandwidth = CLI::GetParam<double>("bandwidth");

    LaplacianKernel kernel(bandwidth);
    KernelPCA<LaplacianKernel> kpca(kernel, scaleData);
    kpca.Apply(dataset, newDim);
  }
  else if (kernelType == "cosine")
  {
    KernelPCA<CosineDistance> kpca(CosineDistance(), scaleData);
    kpca.Apply(dataset, newDim);
  }
  else
  {
    // Invalid kernel type.
    Log::Fatal << "Invalid kernel type ('" << kernelType << "'); valid choices "
        << "are 'linear', 'gaussian', 'polynomial', 'hyptan', 'laplacian', and "
        << "'cosine'." << endl;
  }

  // Save the output dataset.
  const string outputFile = CLI::GetParam<string>("output_file");
  data::Save(outputFile, dataset, true); // Fatal on failure.
}
