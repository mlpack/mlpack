/**
 * @file example_kernel.hpp
 * @author Ryan Curtin
 *
 * This is an example kernel.  If you are making your own kernel, follow the
 * outline specified in this file.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_EXAMPLE_KERNEL_HPP
#define MLPACK_CORE_KERNELS_EXAMPLE_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {

/**
 * @brief Kernel functions.
 *
 * This namespace contains kernel functions, which evaluate some kernel function
 * @f$ K(x, y) @f$ for some arbitrary vectors @f$ x @f$ and @f$ y @f$ of the
 * same dimension.  The single restriction on the function @f$ K(x, y) @f$ is
 * that it must satisfy Mercer's condition:
 *
 * @f[
 * \int \int K(x, y) g(x) g(y) dx dy \ge 0
 * @f]
 *
 * for all square integrable functions @f$ g(x) @f$.
 *
 * The kernels in this namespace all implement the KernelType policy.  For more
 * information, see \ref kernels "The KernelType policy documentation".
 */
namespace kernel {

/**
 * An example kernel function.  This is not a useful kernel, but it implements
 * the two functions necessary to satisfy the Kernel policy (so that a class can
 * be used whenever an mlpack method calls for a `typename Kernel` template
 * parameter.
 *
 * All that is necessary is a constructor and an `Evaluate()` function.  More
 * methods could be added; for instance, one useful idea is a constructor which
 * takes parameters for a kernel (for instance, the width of the Gaussian for a
 * Gaussian kernel).  However, mlpack methods cannot count on these various
 * constructors existing, which is why most methods allow passing an
 * already-instantiated kernel object (and by default the method will construct
 * the kernel with the default constructor).  So, for instance,
 *
 * @code
 * GaussianKernel k(5.0);
 * KernelPCA<GaussianKernel> kpca(dataset, k);
 * @endcode
 *
 * will set up kernel PCA using a Gaussian kernel with a width of 5.0, but
 *
 * @code
 * KernelPCA<GaussianKernel> kpca(dataset);
 * @endcode
 *
 * will create the kernel with the default constructor.  It is important (but
 * not strictly mandatory) that your default constructor still gives a working
 * kernel.
 *
 * @note
 * Not all kernels require state.  For instance, the regular dot product needs
 * no parameters.  In that case, no local variables are necessary and
 * `Evaluate()` can (and should) be declared static.  However, for greater
 * generalization, mlpack methods expect all kernels to require state and hence
 * must store instantiated kernel functions; this is why a default constructor
 * is necessary.
 * @endnote
 */
class ExampleKernel
{
 public:
  /**
   * The default constructor, which takes no parameters.  Because our simple
   * example kernel has no internal parameters that need to be stored, the
   * constructor does not need to do anything.  For a more complex example, see
   * the GaussianKernel, which stores an internal parameter.
   */
  ExampleKernel() { }

  /**
   * Evaluates the kernel function for two given vectors.  In this case, because
   * our simple example kernel has no internal parameters, we can declare the
   * function static.  For a more complex example which cannot be declared
   * static, see the GaussianKernel, which stores an internal parameter.
   *
   * @tparam VecTypeA Type of first vector (arma::vec, arma::sp_vec should be
   *      expected).
   * @tparam VecTypeB Type of second vector (arma::vec, arma::sp_vec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecTypeA, typename VecTypeB>
  static double Evaluate(const VecTypeA& /* a */, const VecTypeB& /* b */)
  { return 0; }

  /**
   * Serializes the kernel.  In this case, the kernel has no members, so we do
   * not need to do anything at all.
   */
  template<typename Archive>
  void Serialize(Archive& /* ar */, const unsigned int /* version */) { }

  /**
   * Obtains the convolution integral [integral K(||x-a||)K(||b-x||)dx]
   * for the two vectors.  In this case, because
   * our simple example kernel has no internal parameters, we can declare the
   * function static.  For a more complex example which cannot be declared
   * static, see the GaussianKernel, which stores an internal parameter.
   *
   * @tparam VecTypeA Type of first vector (arma::vec, arma::sp_vec should be
   *      expected).
   * @tparam VecTypeB Type of second vector (arma::vec, arma::sp_vec).
   * @param a First vector.
   * @param b Second vector.
   * @return the convolution integral value.
   */
  template<typename VecTypeA, typename VecTypeB>
  static double ConvolutionIntegral(const VecTypeA& /* a */,
                                    const VecTypeB& /* b */) { return 0; }

  /**
   * Obtains the normalizing volume for the kernel with dimension $dimension$.
   * In this case, because our simple example kernel has no internal parameters,
   * we can declare the function static.  For a more complex example which
   * cannot be declared static, see the GaussianKernel, which stores an internal
   * parameter.
   *
   * @param dimension the dimension of the space.
   * @return the normalization constant.
   */
  static double Normalizer() { return 0; }

  // Modified to remove unused variable "dimension"
  //static double Normalizer(size_t dimension=1) { return 0; }
};

} // namespace kernel
} // namespace mlpack

#endif
