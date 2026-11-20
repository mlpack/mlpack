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

inline FastMKSModel::FastMKSModel(const int kernelType) :
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

inline FastMKSModel::FastMKSModel(const FastMKSModel& other) :
    kernelType(other.kernelType),
    linear(other.linear == NULL ? NULL :
        new FastMKS<LinearKernel>(*other.linear)),
    polynomial(other.polynomial == NULL ? NULL :
        new FastMKS<PolynomialKernel>(*other.polynomial)),
    cosine(other.cosine == NULL ? NULL :
        new FastMKS<CosineSimilarity>(*other.cosine)),
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

inline FastMKSModel::FastMKSModel(FastMKSModel&& other) :
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

inline FastMKSModel& FastMKSModel::operator=(const FastMKSModel& other)
{
  if (this != &other)
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
      cosine = new FastMKS<CosineSimilarity>(*other.cosine);
    if (other.gaussian)
      gaussian = new FastMKS<GaussianKernel>(*other.gaussian);
    if (other.epan)
      epan = new FastMKS<EpanechnikovKernel>(*other.epan);
    if (other.triangular)
      triangular = new FastMKS<TriangularKernel>(*other.triangular);
    if (other.hyptan)
      hyptan = new FastMKS<HyperbolicTangentKernel>(*other.hyptan);
  }
  return *this;
}

inline FastMKSModel& FastMKSModel::operator=(FastMKSModel&& other)
{
  if (this != &other)
  {
    kernelType = other.kernelType;
    linear = other.linear;
    polynomial = other.polynomial;
    cosine = other.cosine;
    gaussian = other.gaussian;
    epan = other.epan;
    triangular = other.triangular;
    hyptan = other.hyptan;

    // Clear other object.
    other.kernelType = KernelTypes::LINEAR_KERNEL;
    other.linear = nullptr;
    other.polynomial = nullptr;
    other.cosine = nullptr;
    other.gaussian = nullptr;
    other.epan = nullptr;
    other.triangular = nullptr;
    other.hyptan = nullptr;
  }
  return *this;
}

inline FastMKSModel::~FastMKSModel()
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

inline bool FastMKSModel::Naive() const
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->Naive();
    case POLYNOMIAL_KERNEL:
      return polynomial->Naive();
    case COSINE_SIMILARITY:
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

inline bool& FastMKSModel::Naive()
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->Naive();
    case POLYNOMIAL_KERNEL:
      return polynomial->Naive();
    case COSINE_SIMILARITY:
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

inline bool FastMKSModel::SingleMode() const
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->SingleMode();
    case POLYNOMIAL_KERNEL:
      return polynomial->SingleMode();
    case COSINE_SIMILARITY:
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

inline bool& FastMKSModel::SingleMode()
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      return linear->SingleMode();
    case POLYNOMIAL_KERNEL:
      return polynomial->SingleMode();
    case COSINE_SIMILARITY:
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

//! This is called when the KernelType is the same as the model.
template<typename KernelType>
void BuildFastMKSModel(util::Timers& timers,
                       FastMKS<KernelType>& f,
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
    timers.Start("tree_building");
    IPMetric<KernelType> metric(k);
    typename FastMKS<KernelType>::Tree* tree =
        new typename FastMKS<KernelType>::Tree(std::move(referenceData),
                                                metric, base);
    timers.Stop("tree_building");

    f.Train(tree);
  }
}

//! This is only called when something goes wrong.
template<typename KernelType,
         typename FastMKSType>
void BuildFastMKSModel(util::Timers& /* timers */,
                       FastMKSType& /* f */,
                       KernelType& /* k */,
                       arma::mat&& /* referenceData */,
                       const double /* base */)
{
  throw std::invalid_argument("FastMKSModel::BuildModel(): given kernel type is"
      " not equal to kernel type of the model!");
}

template<typename TKernelType>
void FastMKSModel::BuildModel(util::Timers& timers,
                              arma::mat&& referenceData,
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
      linear = new FastMKS<LinearKernel>(singleMode, naive);
      BuildFastMKSModel(timers, *linear, kernel, std::move(referenceData),
          base);
      break;

    case POLYNOMIAL_KERNEL:
      polynomial = new FastMKS<PolynomialKernel>(singleMode, naive);
      BuildFastMKSModel(timers, *polynomial, kernel, std::move(referenceData),
          base);
      break;

    case COSINE_SIMILARITY:
      cosine = new FastMKS<CosineSimilarity>(singleMode, naive);
      BuildFastMKSModel(timers, *cosine, kernel, std::move(referenceData),
          base);
      break;

    case GAUSSIAN_KERNEL:
      gaussian = new FastMKS<GaussianKernel>(singleMode, naive);
      BuildFastMKSModel(timers, *gaussian, kernel, std::move(referenceData),
          base);
      break;

    case EPANECHNIKOV_KERNEL:
      epan = new FastMKS<EpanechnikovKernel>(singleMode, naive);
      BuildFastMKSModel(timers, *epan, kernel, std::move(referenceData), base);
      break;

    case TRIANGULAR_KERNEL:
      triangular = new FastMKS<TriangularKernel>(singleMode, naive);
      BuildFastMKSModel(timers, *triangular, kernel, std::move(referenceData),
          base);
      break;

    case HYPTAN_KERNEL:
      hyptan = new FastMKS<HyperbolicTangentKernel>(singleMode, naive);
      BuildFastMKSModel(timers, *hyptan, kernel, std::move(referenceData),
          base);
      break;
  }
}

template<typename Archive>
void FastMKSModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(kernelType));

  if (cereal::is_loading<Archive>())
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
      ar(CEREAL_POINTER(linear));
      break;

    case POLYNOMIAL_KERNEL:
      ar(CEREAL_POINTER(polynomial));
      break;

    case COSINE_SIMILARITY:
      ar(CEREAL_POINTER(cosine));
      break;

    case GAUSSIAN_KERNEL:
      ar(CEREAL_POINTER(gaussian));
      break;

    case EPANECHNIKOV_KERNEL:
      ar(CEREAL_POINTER(epan));
      break;

    case TRIANGULAR_KERNEL:
      ar(CEREAL_POINTER(triangular));
      break;

    case HYPTAN_KERNEL:
      ar(CEREAL_POINTER(hyptan));
      break;
  }
}

template<typename FastMKSType>
void FastMKSModel::Search(util::Timers& timers,
                          FastMKSType& f,
                          const arma::mat& querySet,
                          const size_t k,
                          arma::Mat<size_t>& indices,
                          arma::mat& kernels,
                          const double base)
{
  if (f.Naive() || f.SingleMode())
  {
    timers.Start("computing_products");
    f.Search(querySet, k, indices, kernels);
    timers.Stop("computing_products");
  }
  else
  {
    timers.Start("tree_building");
    typename FastMKSType::Tree queryTree(querySet, base);
    timers.Stop("tree_building");

    timers.Start("computing_products");
    f.Search(&queryTree, k, indices, kernels);
    timers.Stop("computing_products");
  }
}

inline void FastMKSModel::Search(
    util::Timers& timers,
    const arma::mat& querySet,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels,
    const double base)
{
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      Search(timers, *linear, querySet, k, indices, kernels, base);
      break;
    case POLYNOMIAL_KERNEL:
      Search(timers, *polynomial, querySet, k, indices, kernels, base);
      break;
    case COSINE_SIMILARITY:
      Search(timers, *cosine, querySet, k, indices, kernels, base);
      break;
    case GAUSSIAN_KERNEL:
      Search(timers, *gaussian, querySet, k, indices, kernels, base);
      break;
    case EPANECHNIKOV_KERNEL:
      Search(timers, *epan, querySet, k, indices, kernels, base);
      break;
    case TRIANGULAR_KERNEL:
      Search(timers, *triangular, querySet, k, indices, kernels, base);
      break;
    case HYPTAN_KERNEL:
      Search(timers, *hyptan, querySet, k, indices, kernels, base);
      break;
    default:
      throw std::runtime_error("invalid model type");
  }
}

inline void FastMKSModel::Search(
    util::Timers& timers,
    const size_t k,
    arma::Mat<size_t>& indices,
    arma::mat& kernels)
{
  timers.Start("computing_products");
  switch (kernelType)
  {
    case LINEAR_KERNEL:
      linear->Search(k, indices, kernels);
      break;
    case POLYNOMIAL_KERNEL:
      polynomial->Search(k, indices, kernels);
      break;
    case COSINE_SIMILARITY:
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
  timers.Stop("computing_products");
}

} // namespace mlpack

#endif
