/**
 * @file steepest_descent.hpp
 * @author Marcus Edel
 *
 * Implementation of the steepest descent optimizer. The method of steepest
 * descent, also called the gradient descent method, is used to find the
 * nearest local minimum of a function which the assumtion that the gradient of
 * the function can be computed.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_STEEPEST_DESCENT_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_STEEPEST_DESCENT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to update the weights using steepest descent.
 *
 * @tparam DataType Type of input data (should be arma::mat,
 * arma::spmat or arma::cube).
 */
template<typename DecomposableFunctionType, typename DataType>
class SteepestDescent
{
 public:
  /**
   * Construct the SteepestDescent optimizer with the given function and
   * parameters.
   *
   * @param function Function to be optimized (minimized).
   * @param lr The learning rate coefficient.
   * @param mom The momentum coefficient.
   */
  SteepestDescent(DecomposableFunctionType& function,
                  const double lr = 0.5,
                  const double mom = 0) :
      function(function),
      lr(lr),
      mom(mom)

  {
    // Nothing to do here.
  }

  /**
   * Optimize the given function using steepest descent.
   */
  void Optimize()
  {
    if (momWeights.n_elem == 0)
    {
      momWeights = function.Weights();
      momWeights.zeros();
    }

    Optimize(function.Weights(), gradient, momWeights);
  }

  /*
   * Sum up all gradients and store the results in the gradients storage.
   */
  void Update()
  {
    if (gradient.n_elem != 0)
    {
      DataType outputGradient = function.Gradient();
      gradient += outputGradient;
    }
    else
    {
      gradient = function.Gradient();
    }
  }

  /*
   * Reset the gradient storage.
   */
  void Reset()
  {
    gradient.zeros();
  }

 private:
  /** Optimize the given function using steepest descent.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param gradient The moving average over the root mean squared gradient used
   *    to update the weights.
   */
  template<typename eT>
  void Optimize(arma::Cube<eT>& weights,
                arma::Cube<eT>& gradient,
                arma::Cube<eT>& momWeights)
  {
    for (size_t s = 0; s < weights.n_slices; s++)
      Optimize(weights.slice(s), gradient.slice(s), momWeights.slice(s));
  }

  /**
   * Optimize the given function using steepest descent.
   *
   * @param weights The weights that should be updated.
   * @param gradient The gradient used to update the weights.
   * @param gradient The moving average over the root mean squared gradient used
   *    to update the weights.
   */
  template<typename eT>
  void Optimize(arma::Mat<eT>& weights,
                arma::Mat<eT>& gradient,
                arma::Mat<eT>& momWeights)
  {
    if (mom > 0)
    {
      momWeights *= mom;
      momWeights += (lr * gradient);
      weights -= momWeights;
    }
    else
    {
      weights -= lr * gradient;
    }
  }

  //! The instantiated function.
  DecomposableFunctionType& function;

  //! The value used as learning rate.
  const double lr;

  //! The value used as momentum.
  const double mom;

  //! Momentum matrix.
  DataType momWeights;

  //! The current gradient.
  DataType gradient;
}; // class SteepestDescent

} // namespace ann
} // namespace mlpack

#endif
