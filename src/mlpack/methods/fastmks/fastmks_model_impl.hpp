/**
 * @file methods/fastmks/fastmks_model_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated functions of FastMKSModel.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_FASTMKS_MODEL_IMPL_HPP
#define MLPACK_METHODS_FASTMKS_FASTMKS_MODEL_IMPL_HPP

#include "fastmks_model.hpp"

namespace mlpack {
namespace fastmks {

//! This is called when the KernelType is the same as the model.
template<typename KernelType>
void BuildFastMKSModel(FastMKS<KernelType>& f,
                       KernelType& k,
                       arma::mat&& referenceData,
                       const double base)
{
  // Do we need to build the tree?
  if (base <= 1.0)
  {
    throw std::invalid_argument("base must be greater than 1");
  }

  if (f.Naive())
  {
    f.Train(std::move(referenceData), k);
  }
  else
  {
    // Create the tree with the specified base.
    Timer::Start("tree_building");
    metric::IPMetric<KernelType> metric(k);
    typename FastMKS<KernelType>::Tree* tree =
        new typename FastMKS<KernelType>::Tree(std::move(referenceData),
                                                metric, base);
    Timer::Stop("tree_building");

    f.Train(tree);
  }
}

//! This is only called when something goes wrong.
template<typename KernelType,
         typename FastMKSType>
void BuildFastMKSModel(FastMKSType& /* f */,
                       KernelType& /* k */,
                       arma::mat&& /* referenceData */,
                       const double /* base */)
{
  throw std::invalid_argument("FastMKSModel::BuildModel(): given kernel type is"
      " not equal to kernel type of the model!");
}

template<typename TKernelType>
void FastMKSModel::BuildModel(arma::mat&& referenceData,
                              TKernelType& kernel,
                              const bool singleMode,
                              const bool naive,
                              const double base)
{
  // Clean memory if necessary.
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

  linear = NULL;
  polynomial = NULL;
  cosine = NULL;
  gaussian = NULL;
  epan = NULL;
  triangular = NULL;
  hyptan = NULL;

  // Instantiate the right model.
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      linear = new FastMKS<kernel::LinearKernel>(singleMode, naive);
      BuildFastMKSModel(*linear, kernel, std::move(referenceData), base);
      break;

    case POLYNOMIAL_KERNEL:
      polynomial = new FastMKS<kernel::PolynomialKernel>(singleMode, naive);
      BuildFastMKSModel(*polynomial, kernel, std::move(referenceData), base);
      break;

    case COSINE_DISTANCE:
      cosine = new FastMKS<kernel::CosineDistance>(singleMode, naive);
      BuildFastMKSModel(*cosine, kernel, std::move(referenceData), base);
      break;

    case GAUSSIAN_KERNEL:
      gaussian = new FastMKS<kernel::GaussianKernel>(singleMode, naive);
      BuildFastMKSModel(*gaussian, kernel, std::move(referenceData), base);
      break;

    case EPANECHNIKOV_KERNEL:
      epan = new FastMKS<kernel::EpanechnikovKernel>(singleMode, naive);
      BuildFastMKSModel(*epan, kernel, std::move(referenceData), base);
      break;

    case TRIANGULAR_KERNEL:
      triangular = new FastMKS<kernel::TriangularKernel>(singleMode, naive);
      BuildFastMKSModel(*triangular, kernel, std::move(referenceData), base);
      break;

    case HYPTAN_KERNEL:
      hyptan = new FastMKS<kernel::HyperbolicTangentKernel>(singleMode, naive);
      BuildFastMKSModel(*hyptan, kernel, std::move(referenceData), base);
      break;
  }
}

template<typename Archive>
void FastMKSModel::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  ar & CEREAL_NVP(kernelType);

  if (Archive::is_loading::value)
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

    linear = NULL;
    polynomial = NULL;
    cosine = NULL;
    gaussian = NULL;
    epan = NULL;
    triangular = NULL;
    hyptan = NULL;
  }

  // Serialize the correct model.
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      ar & CEREAL_POINTER(linear);
      break;

    case POLYNOMIAL_KERNEL:
      ar & CEREAL_POINTER(polynomial);
      break;

    case COSINE_DISTANCE:
      ar & CEREAL_POINTER(cosine);
      break;

    case GAUSSIAN_KERNEL:
      ar & CEREAL_POINTER(gaussian);
      break;

    case EPANECHNIKOV_KERNEL:
      ar & CEREAL_POINTER(epan);
      break;

    case TRIANGULAR_KERNEL:
      ar & CEREAL_POINTER(triangular);
      break;

    case HYPTAN_KERNEL:
      ar & CEREAL_POINTER(hyptan);
      break;
  }
}

template<typename FastMKSType>
void FastMKSModel::Search(FastMKSType& f,
                          const arma::mat& querySet,
                          const size_t k,
                          arma::Mat<size_t>& indices,
                          arma::mat& kernels,
                          const double base)
{
  if (f.Naive() || f.SingleMode())
  {
    f.Search(querySet, k, indices, kernels);
  }
  else
  {
    Timer::Start("tree_building");
    typename FastMKSType::Tree queryTree(querySet, base);
    Timer::Stop("tree_building");

    f.Search(&queryTree, k, indices, kernels);
  }
}

} // namespace fastmks
} // namespace mlpack

#endif
