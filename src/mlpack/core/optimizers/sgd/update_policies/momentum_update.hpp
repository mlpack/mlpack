/**
 * @file momentum_update.hpp
 * @author Arun Reddy
 *
 * Momentum update for Stochastic Gradient Descent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_MOMENTUM_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_MOMENTUM_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Momentum update policy for Stochastic Gradient Descent (SGD).
 *
 * Learning with SGD can sometimes be slow.  Using momentum update for parameter
 * learning can accelerate the rate of convergence, specifically in the cases
 * where the surface curves much more steeply(a steep hilly terrain with high
 * curvature).  The momentum algorithm introduces a new velocity vector \f$ v
 * \f$ with the same dimension as the parameter \f$ A \f$.  Also it introduces a
 * new decay hyperparameter momentum \f$ mu \in (0,1] \f$.  The following update
 * scheme is used to update SGD in every iteration:
 *
 * \f[
 * v = mu*v - \alpha \nabla f_i(A)
 * A_{j + 1} = A_j + v
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).  Common values of
 * \f$ mu \f$ include 0.5, 0.9 and 0.99.  Typically it begins with a small value
 * and later raised.
 *
 * For more information, please refer to the Section 8.3.2 of the following book
 *
 * @code
 * @article{rumelhart1988learning,
 *   title   = {Learning representations by back-propagating errors},
 *   author  = {Rumelhart, David E. and Hinton, Geoffrey E. and
 *              Williams, Ronald J.},
 *   journal = {Cognitive Modeling},
 *   volume  = {5},
 *   number  = {3},
 *   pages   = {1},
 *   year    = {1988}
 * }
 *
 * @code
 * @book{Goodfellow-et-al-2016,
 *  title     = {Deep Learning},
 *  author    = {Ian Goodfellow and Yoshua Bengio and Aaron Courville},
 *  publisher = {MIT Press},
 *  note      = {\url{http://www.deeplearningbook.org}},
 *  year      = {2016}
 * }
 */
class MomentumUpdate
{
 public:
  /**
   * Construct the momentum update policy with given momentum decay parameter.
   *
   * @param momentum The momentum decay hyperparameter
   */
  MomentumUpdate(const double momentum = 0.5) : momentum(momentum)
  { /* Do nothing. */ };

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.  In the momentum update policy the velocity
   * matrix is initialized to the zeros matrix with the same size as the
   * gradient matrix (see mlpack::optimization::SGD::Optimizer )
   *
   * @param rows Number of rows in the gradient matrix.
   * @param cols Number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows, const size_t cols)
  {
    // Initialize am empty velocity matrix.
    velocity = arma::zeros<arma::mat>(rows, cols);
  }

  /**
   * Update step for SGD.  The momentum term makes the convergence faster on the
   * way as momentum term increases for dimensions pointing in the same and
   * reduces updates for dimensions whose gradients change directions.
   *
   * @param iterate Parameters that minimize the function.
   * @param stepSize Step size to be used for the given iteration.
   * @param gradient The gradient matrix.
   */
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient)
  {
    velocity = momentum * velocity - stepSize * gradient;
    iterate += velocity;
  }

 private:
  // The momentum hyperparamter
  double momentum;
  // The velocity matrix.
  arma::mat velocity;
};

} // namespace optimization
} // namespace mlpack

#endif
