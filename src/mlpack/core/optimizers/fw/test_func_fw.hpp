/**
 * @file test_func_sq.hpp
 * @author Chenzhe Diao
 *
 * Simple test function for classic Frank Wolfe Algorithm:
 *
 * \f$ f(x) = (x1 - 0.1)^2 + (x2 - 0.2)^2 + (x3 - 0.3)^2 \f$
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_TEST_FUNC_FW_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_TEST_FUNC_FW_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Simple test function for classic Frank Wolfe Algorithm:
 *
 * \f$ f(x) = (x1 - 0.1)^2 + (x2 - 0.2)^2 + (x3 - 0.3)^2 \f$.
 */
class TestFuncFW
{
 public:
  TestFuncFW()
  {/* Nothing to do. */}

  /**
   * Evaluation of the function.
   *
   * @param coords input vector x.
   */
  double Evaluate(const arma::mat& coords)
  {
    double f = std::pow(coords[0] - 0.1, 2);
    f += std::pow(coords[1] - 0.2, 2);
    f += std::pow(coords[2] - 0.3, 2);
    return f;
  }

  /**
   * Gradient of the function.
   *
   * @param coords input vector x.
   * @param gradient output gradient vector.
   */
  void Gradient(const arma::mat& coords, arma::mat& gradient)
  {
    gradient.set_size(3, 1);
    gradient[0] = coords[0] - 0.1;
    gradient[1] = coords[1] - 0.2;
    gradient[2] = coords[2] - 0.3;
  }
};

}  // namespace optimization
}  // namespace mlpack

#endif
