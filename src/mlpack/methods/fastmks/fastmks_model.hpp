/**
 * @file methods/fastmks/fastmks_model.hpp
 * @author Ryan Curtin
 *
 * A utility struct to contain all the possible FastMKS models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_FASTMKS_FASTMKS_MODEL_HPP
#define MLPACK_METHODS_FASTMKS_FASTMKS_MODEL_HPP

#include <mlpack/prereqs.hpp>
#include "fastmks.hpp"
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/cosine_similarity.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/kernels/triangular_kernel.hpp>

namespace mlpack {

//! A utility struct to contain all the possible FastMKS models, for use by the
//! mlpack_fastmks program.
class FastMKSModel
{
 public:
  //! A list of all the kernels we support.
  enum KernelTypes
  {
    LINEAR_KERNEL,
    POLYNOMIAL_KERNEL,
    COSINE_SIMILARITY,
    GAUSSIAN_KERNEL,
    EPANECHNIKOV_KERNEL,
    TRIANGULAR_KERNEL,
    HYPTAN_KERNEL
  };

  /**
   * Create the FastMKSModel with the given kernel type.
   */
  FastMKSModel(const int kernelType = LINEAR_KERNEL);

  //! Copy constructor.
  FastMKSModel(const FastMKSModel& other);

  //! Move constructor.
  FastMKSModel(FastMKSModel&& other);

  //! Copy assignment operator.
  FastMKSModel& operator=(const FastMKSModel& other);

  //! Move assignment operator.
  FastMKSModel& operator=(FastMKSModel&& other);

  /**
   * Clean memory.
   */
  ~FastMKSModel();

  /**
   * Build the model on the given reference set.  Make sure kernelType is equal
   * to the correct entry in KernelTypes for the given KernelType class!
   */
  template<typename TKernelType>
  void BuildModel(util::Timers& timers,
                  arma::mat&& referenceData,
                  TKernelType& kernel,
                  const bool singleMode,
                  const bool naive,
                  const double base);

  //! Get whether or not naive search is used.
  bool Naive() const;
  //! Set whether or not naive search is used.
  bool& Naive();

  //! Get whether or not single-tree search is used.
  bool SingleMode() const;
  //! Set whether or not single-tree search is used.
  bool& SingleMode();

  //! Get the kernel type.
  int KernelType() const { return kernelType; }
  //! Modify the kernel type.
  int& KernelType() { return kernelType; }

  /**
   * Search with a different query set.
   *
   * @param querySet Set to search with.
   * @param k Number of max-kernel candidates to search for.
   * @param indices A matrix in which to store the indices of max-kernel
   *      candidates.
   * @param kernels A matrix in which to store the max-kernel candidate kernel
   *      values.
   * @param base Base to use for cover tree building (if in dual-tree search
   *      mode).
   */
  void Search(util::Timers& timers,
              const arma::mat& querySet,
              const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& kernels,
              const double base);

  /**
   * Search with the reference set as the query set.
   *
   * @param k Number of max-kernel candidates to search for.
   * @param indices A matrix in which to store the indices of max-kernel
   *      candidates.
   * @param kernels A matrix in which to store the max-kernel candidate kernel
   *      values.
   */
  void Search(util::Timers& timers,
              const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& kernels);

  /**
   * Serialize the model.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! The type of kernel we are using.
  int kernelType;

  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<LinearKernel>* linear;
  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<PolynomialKernel>* polynomial;
  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<CosineSimilarity>* cosine;
  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<GaussianKernel>* gaussian;
  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<EpanechnikovKernel>* epan;
  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<TriangularKernel>* triangular;
  //! This will only be non-NULL if this is the type of kernel we are using.
  FastMKS<HyperbolicTangentKernel>* hyptan;

  //! Build a query tree and execute the search.
  template<typename FastMKSType>
  void Search(util::Timers& timers,
              FastMKSType& f,
              const arma::mat& querySet,
              const size_t k,
              arma::Mat<size_t>& indices,
              arma::mat& kernels,
              const double base);
};

} // namespace mlpack

#include "fastmks_model_impl.hpp"

#endif
