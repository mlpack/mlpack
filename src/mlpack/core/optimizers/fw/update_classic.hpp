/**
 * @file update_classic.hpp
 * @author Chenzhe Diao
 *
 * Classic update method for FrankWolfe algorithm. Used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_CLASSIC_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_CLASSIC_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Use classic rule in the update step for FrankWolfe algorithm. That is,
 * take \f$ \gamma = \frac{2}{k+2} \f$, where \f$ k \f$ is the iteration 
 * number. The update rule would be:
 * \f[
 * x_{k+1} = (1-\gamma) x_k + \gamma s
 * \f]
 *
 */
class UpdateClassic
{
 public:
  /**
   * Construct the classic update rule for FrankWolfe algorithm.
   */
  UpdateClassic()
  { /* Do nothing. */ }

  /**
   * Classic update rule for FrankWolfe.
   *
   * \f$ x_{k+1} = (1-\gamma)x_k + \gamma s \f$, where \f$ \gamma = 2/(k+2) \f$
   *
   * @param old_coords previous solution coords.
   * @param s current linear_constr_solution result.
   * @param new_coords new output solution coords.
   * @param num_iter current iteration number
   */
  template<typename FunctionType>
  void Update(FunctionType& function,
              const arma::mat& old_coords,
              const arma::mat& s,
              arma::mat& new_coords,
              const size_t num_iter)
  {
    double gamma = 2.0 / (num_iter + 2.0);
    new_coords = (1.0 - gamma) * old_coords + gamma * s;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
