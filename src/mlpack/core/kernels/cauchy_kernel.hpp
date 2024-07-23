/**
 * @file core/kernels/cauchy_kernel.hpp
 * @author Tommi Laivamaa
 *
 * Implementation of the Cauchy kernel (CauchyKernel),
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_CAUCHY_KERNEL_HPP
#define MLPACK_CORE_KERNELS_CAUCHY_KERNEL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/distances/lmetric.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>

namespace mlpack {

/**
 * The Cauchy kernel. Given two vector @f$ x @f$, @f$ y @f$, and a bandwidth
 * @f$ \sigma @f$ (set in the constructor),
 *
 * @f[
 * K(x, y) = \frac{1}{1 + (\frac{|| x - y ||}{\sigma})^2}.
 * @f]
 *
 * For more details, see the following published paper:
 *
 * @code
 * @inproceedings{Basak2008,
 *   title={A least square kernel machine with box constraints},
 *   author={Basak, Jayanta},
 *   booktitle={Pattern Recognition, 2008. ICPR 2008. 19th International
 *       Conference on},
 *   pages={1--4},
 *   year={2008},
 *   organization={IEEE}
 * }
 * @endcode
 */
class CauchyKernel
{
 public:
  /**
   * Construct the Cauchy kernel; by default, the bandwidth is 1.0.
   */
  CauchyKernel(double bandwidth = 1.0) : bandwidth(bandwidth)
  { }

  /**
   * Evaluation of the Cauchy kernel. This could be generalized to use any
   * distance metric, not the Euclidean distance, but for now, the Euclidean
   * distance is used.
   *
   * @tparam VecTypeA Type of first vector (arma::vec, arma::sp_vec).
   * @tparam VecTypeB Type of second vector (arma::vec, arma::sp_vec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b)
  {
    return 1 / (1 + (
        std::pow(EuclideanDistance::Evaluate(a, b) / bandwidth, 2)));
  }

  /**
   * Serialize the kernel.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(bandwidth));
  }

  // Get the kernel bandwidth.
  double Bandwidth() const { return bandwidth; }
  // Modify the kernel bandwidth.
  void Bandwidth(const double bw) { this->bandwidth = bw; }

 private:
  //! Kernel bandwidth.
  double bandwidth;
};

//! Kernel traits for the Cauchy kernel.
template<>
class KernelTraits<CauchyKernel>
{
 public:
  //! The Cauchy kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
};

} // namespace mlpack

#endif
