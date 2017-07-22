/**
 * @file gradient_clipping.hpp
 * @author Konstantin Sidorov
 *
 * Gradient clipping update wrapper.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_GRADIENT_CLIPPING_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_GRADIENT_CLIPPING_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Interface for wrapping around update policies (e.g., VanillaUpdate)
 * and feeding a clipped gradient to them instead of the normal one.
 * (Clipping here is implemented as
 * \f$ g_{\text{clipped}} = \max(g_{\text{min}}, \min(g_{\text{min}}, g))) \f$.)
 * 
 * @tparam UpdatePolicy A type of UpdatePolicy that sould be wrapped around.
 */
template<typename UpdatePolicy>
class GradientClipping
{
 public:
  /**
   * Constructor for creating a GradientClipping instance.
   * 
   * @param minGradient Minimum possible value of gradient element.
   * @param maxGradient Maximum possible value of gradient element.
   * @param updatePolicy An instance of the UpdatePolicy
   *                     used for actual optimization.
   */
  GradientClipping(const double minGradient,
                   const double maxGradient,
                   UpdatePolicy updatePolicy) :
    minGradient(minGradient),
    maxGradient(maxGradient),
    updatePolicy(updatePolicy)
  {
    // Nothing to do here
  }

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process. Here we just do whatever initialization
   * is needed for the actual update policy.
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    updatePolicy.Initialize(rows, cols);
  }

  /**
   * Update step. First, the gradient is clipped, and then the actual update
   * policy does whatever update it needs to do.
   * 
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    // First, clip the gradient.
    gradient.transform(
        [&](double val)
        { return std::min(std::max(val, minGradient), maxGradient); });
    // And only then do the update.
    updatePolicy.Update(iterate, stepSize, gradient);
  }
 private:
  //! Minimum possible value of gradient element.
  double minGradient;

  //! Maximum possible value of gradient element.
  double maxGradient;

  //! An instance of the UpdatePolicy used for actual optimization.
  UpdatePolicy updatePolicy;
};

} // namespace optimization
} // namespace mlpack

#endif
