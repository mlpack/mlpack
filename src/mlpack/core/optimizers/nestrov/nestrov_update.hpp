/**
 * @file nestrov_update.hpp
 * @author Kris Singh
 *
 * Nestrov Momentum update for Stochastic Gradient Descent.
 * 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SGD_NESTROV_UPDATE_HPP
#define MLPACK_CORE_OPTIMIZERS_SGD_NESTROV_UPDATE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Construct the NAG optimizer with the given function and parameters.  The
 * defaults here are not necessarily good for the given problem, so it is
 * suggested that the values used be tailored to the task at hand.  The
 * maximum number of iterations refers to the maximum number of points that
 * are processed (i.e., one iteration equals one point; one iteration does not
 * equal one pass over the dataset). Typically the momentum paramter is often
 * initialized with small value like 0.5 and later raised.

 * @misc{1212.0901,
 * Author = {Yoshua Bengio and Nicolas Boulanger-Lewandowski and Razvan Pascanu},
 * Title = {Advances in Optimizing Recurrent Networks},
 * Year = {2012},
 * Eprint = {arXiv:1212.0901},
 * } 
 *
 * @code
 * @book{Goodfellow-et-al-2016,
 *  title={Deep Learning},
 *  author={Ian Goodfellow and Yoshua Bengio and Aaron Courville},
 *  publisher={MIT Press},
 *  note={\url{http://www.deeplearningbook.org}},
 *  year={2016}
 * }
 */
class NestrovUpdate
{
 public:
  /**
   * Construct the momentum update policy with given momentum decay parameter.
   *
   * @param momentum The momentum decay hyperparameter
   */
  NestrovUpdate(const double momentum = 0.5) : momentum(momentum)
  { /* Do nothing. */ };

  /**
   * The Initialize method is called by SGD Optimizer method before the start of
   * the iteration update process.  In the momentum update policy the velocity
   * matrix is initialized to the zeros matrix with the same size as the
   * gradient matrix (see mlpack::optimization::SGD::Optimizer )
   *
   * @param n_rows number of rows in the gradient matrix.
   * @param n_cols number of columns in the gradient matrix.
   */
  void Initialize(const size_t rows,
                  const size_t cols)
  {
    //Initialize am empty velocity matrix.
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
  template<typename... T>
  void Update(arma::mat& iterate,
              const double stepSize,
              const arma::mat& gradient,
              T... args)
  {
    velocity = momentum * velocity - stepSize * gradient;
    iterate += momentumLookAhead(momentum) * momentum * velocity 
      - (1 + momentumLookAhead(momentum)) * stepSize * gradient;
  }

  double momentumLookAhead(double momentum)
  {
    /*Implement a momentum decay scheme*/
    return momentum;
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
