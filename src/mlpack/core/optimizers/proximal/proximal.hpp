/**
 * @file proximal.hpp
 * @author Chenzhe Diao
 *
 * Approximate a vector with another vector on lp ball. Currently support l0
 * ball and l1 ball with specific norm.
 * It can be used in projected gradient method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_PROXIMAL_PROXIMAL_HPP
#define MLPACK_CORE_OPTIMIZERS_PROXIMAL_PROXIMAL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Approximate a vector with another vector on lp ball. Currently support l0
 * ball and l1 ball with specific norm.
 * It can be used in projected gradient method.
 */
class Proximal
{
 public:
  /**
   * Project the vector onto the l1 ball with norm tau. That is, we will solve
   * for:
   * \f[
   * w = argmin_w ||w - v||_2, \qquad s.t. ~ ||w||_1 \leqslant tau
   * \f]
   *
   * @param v Input vector to be approxmated, the output optimal vector is
   *          also saved in v.
   * @param tau Norm of l1 ball.
   */
  static void ProjectToL1Ball(arma::vec& v, double tau);

  /**
   * Project the vector onto the l0 ball with norm tau. That is, we try to
   * approximate v with sparse vector w:
   * \f[
   * w = argmin_w ||w - v||_2, \qquad s.t. ~ ||w||_0 \leqslant tau
   * \f]
   *
   * @param v Input vector to be approxmated, the output optimal vector is
   *          also saved in v.
   * @param tau Norm of l0 ball.
   */
  static void ProjectToL0Ball(arma::vec& v, int tau);
};  // class Proximal

} // namespace optimization
} // namespace mlpack

#endif
