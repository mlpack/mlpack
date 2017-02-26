/**
 * @file fastmks_model.cpp
 * @author Ryan Curtin
 *
 * Implementation of non-templatized functions of FastMKSModel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "fastmks_model.hpp"

using namespace mlpack;
using namespace mlpack::fastmks;
using namespace mlpack::kernel;

FastMKSModel::FastMKSModel(const int kernelType) :
    kernelType(kernelType),
    linear(NULL),
    polynomial(NULL),
    cosine(NULL),
    gaussian(NULL),
    epan(NULL),
    triangular(NULL),
    hyptan(NULL)
{
  // Nothing to do.
}

FastMKSModel::FastMKSModel(const FastMKSModel& other) :
    kernelType(other.kernelType),
    linear(other.linear == NULL ? NULL :
        new FastMKS<LinearKernel>(*other.linear)),
    polynomial(other.polynomial == NULL ? NULL :
        new FastMKS<PolynomialKernel>(*other.polynomial)),
    cosine(other.cosine == NULL ? NULL :
        new FastMKS<CosineDistance>(*other.cosine)),
    gaussian(other.gaussian == NULL ? NULL :
        new FastMKS<GaussianKernel>(*other.gaussian)),
    epan(other.epan == NULL ? NULL :
        new FastMKS<EpanechnikovKernel>(*other.epan)),
    triangular(other.triangular == NULL ? NULL :
        new FastMKS<TriangularKernel>(*other.triangular)),
    hyptan(other.hyptan == NULL ? NULL :
        new FastMKS<HyperbolicTangentKernel>(*other.hyptan))
{
  // Nothing to do.
}

FastMKSModel::FastMKSModel(FastMKSModel&& other) :
    kernelType(other.kernelType),
    linear(other.linear),
    polynomial(other.polynomial),
    cosine(other.cosine),
    gaussian(other.gaussian),
    epan(other.epan),
    triangular(other.triangular),
    hyptan(other.hyptan)
{
  // Clear other object.
  other.kernelType = KernelTypes::LINEAR_KERNEL;
  other.linear = NULL;
  other.polynomial = NULL;
  other.cosine = NULL;
  other.gaussian = NULL;
  other.epan = NULL;
  other.triangular = NULL;
  other.hyptan = NULL;
}

FastMKSModel& FastMKSModel::operator=(const FastMKSModel& other)
{
  // Clear memory.
  delete linear;
  delete polynomial;
  delete cosine;
  delete gaussian;
  delete epan;
  delete triangular;
  delete hyptan;

  // Set pointers to null.
  linear = NULL;
  polynomial = NULL;
  cosine = NULL;
  gaussian = NULL;
  epan = NULL;
  triangular = NULL;
  hyptan = NULL;

  kernelType = other.kernelType;
  if (other.linear)
    linear = new FastMKS<LinearKernel>(*other.linear);
  if (other.polynomial)
    polynomial = new FastMKS<PolynomialKernel>(*other.polynomial);
  if (other.cosine)
    cosine = new FastMKS<CosineDistance>(*other.cosine);
  if (other.gaussian)
    gaussian = new FastMKS<GaussianKernel>(*other.gaussian);
  if (other.epan)
    epan = new FastMKS<EpanechnikovKernel>(*other.epan);
  if (other.triangular)
    triangular = new FastMKS<TriangularKernel>(*other.triangular);
  if (other.hyptan)
    hyptan = new FastMKS<HyperbolicTangentKernel>(*other.hyptan);

  return *this;
}

FastMKSModel::~FastMKSModel()
{
  // Clean memory.
  if (linear)
    delete linear;
  if (polynomial)
    delete polynomial;
  if (cosine)
    delete cosine;
  if (gaussian)
    delete gaussian;
  if (epan)
    delete epan;
  if (triangular)
    delete triangular;
  if (hyptan)
    delete hyptan;
}

bool FastMKSModel::Naive() const
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->Naive();
    case POLYNOMIAL_KERNEL:
      return polynomial->Naive();
    case COSINE_DISTANCE:
      return cosine->Naive();
    case GAUSSIAN_KERNEL:
      return gaussian->Naive();
    case EPANECHNIKOV_KERNEL:
      return epan->Naive();
    case TRIANGULAR_KERNEL:
      return triangular->Naive();
    case HYPTAN_KERNEL:
      return hyptan->Naive();
  }

  throw std::runtime_error("invalid model type");
}

bool& FastMKSModel::Naive()
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->Naive();
    case POLYNOMIAL_KERNEL:
      return polynomial->Naive();
    case COSINE_DISTANCE:
      return cosine->Naive();
    case GAUSSIAN_KERNEL:
      return gaussian->Naive();
    case EPANECHNIKOV_KERNEL:
      return epan->Naive();
    case TRIANGULAR_KERNEL:
      return triangular->Naive();
    case HYPTAN_KERNEL:
      return hyptan->Naive();
  }

  throw std::runtime_error("invalid model type");
}

bool FastMKSModel::SingleMode() const
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->SingleMode();
    case POLYNOMIAL_KERNEL:
      return polynomial->SingleMode();
    case COSINE_DISTANCE:
      return cosine->SingleMode();
    case GAUSSIAN_KERNEL:
      return gaussian->SingleMode();
    case EPANECHNIKOV_KERNEL:
      return epan->SingleMode();
    case TRIANGULAR_KERNEL:
      return triangular->SingleMode();
    case HYPTAN_KERNEL:
      return hyptan->SingleMode();
  }

  throw std::runtime_error("invalid model type");
}

bool& FastMKSModel::SingleMode()
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->SingleMode();
    case POLYNOMIAL_KERNEL:
      return polynomial->SingleMode();
    case COSINE_DISTANCE:
      return cosine->SingleMode();
    case GAUSSIAN_KERNEL:
      return gaussian->SingleMode();
    case EPANECHNIKOV_KERNEL:
      return epan->SingleMode();
    case TRIANGULAR_KERNEL:
      return triangular->SingleMode();
    case HYPTAN_KERNEL:
      return hyptan->SingleMode();
  }

  throw std::runtime_error("invalid model type");
}

void FastMKSModel::Search(const arma::mat& querySet,
                          const size_t k,
                          arma::Mat<size_t>& indices,
                          arma::mat& kernels,
                          const double base)
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      Search(*linear, querySet, k, indices, kernels, base);
      break;
    case POLYNOMIAL_KERNEL:
      Search(*polynomial, querySet, k, indices, kernels, base);
      break;
    case COSINE_DISTANCE:
      Search(*cosine, querySet, k, indices, kernels, base);
      break;
    case GAUSSIAN_KERNEL:
      Search(*gaussian, querySet, k, indices, kernels, base);
      break;
    case EPANECHNIKOV_KERNEL:
      Search(*epan, querySet, k, indices, kernels, base);
      break;
    case TRIANGULAR_KERNEL:
      Search(*triangular, querySet, k, indices, kernels, base);
      break;
    case HYPTAN_KERNEL:
      Search(*hyptan, querySet, k, indices, kernels, base);
      break;
    default:
      throw std::runtime_error("invalid model type");
  }
}

void FastMKSModel::Search(const size_t k,
                          arma::Mat<size_t>& indices,
                          arma::mat& kernels)
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      linear->Search(k, indices, kernels);
      break;
    case POLYNOMIAL_KERNEL:
      polynomial->Search(k, indices, kernels);
      break;
    case COSINE_DISTANCE:
      cosine->Search(k, indices, kernels);
      break;
    case GAUSSIAN_KERNEL:
      gaussian->Search(k, indices, kernels);
      break;
    case EPANECHNIKOV_KERNEL:
      epan->Search(k, indices, kernels);
      break;
    case TRIANGULAR_KERNEL:
      triangular->Search(k, indices, kernels);
      break;
    case HYPTAN_KERNEL:
      hyptan->Search(k, indices, kernels);
      break;
    default:
      throw std::invalid_argument("invalid model type");
  }
}
