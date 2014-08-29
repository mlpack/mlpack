/**
 * @file sparse_autoencoder_function.hpp
 * @author Siddharth Agrawal
 *
 * The function to be optimized for sparse autoencoders.  Any mlpack optimizer
 * can be used.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_FUNCTION_HPP
#define __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace nn {

/**
 * This is a class for the sparse autoencoder objective function. It can be used
 * to create learning models like self-taught learning, stacked autoencoders,
 * conditional random fields (CRFs), and so forth.
 */
class SparseAutoencoderFunction
{
 public:
  /**
   * Construct the sparse autoencoder objective function with the given
   * parameters.
   *
   * @param data The data matrix.
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoderFunction(const arma::mat& data,
                            const size_t visibleSize,
                            const size_t hiddenSize,
                            const double lambda = 0.0001,
                            const double beta = 3,
                            const double rho = 0.01);

  //! Initializes the parameters of the model to suitable values.
  const arma::mat InitializeWeights();

  /**
   * Evaluates the objective function of the sparse autoencoder model using the
   * given parameters. The cost function has terms for the reconstruction
   * error, regularization cost and the sparsity cost. The objective function
   * takes a low value when the model is able to reconstruct the data well
   * using weights which are low in value and when the average activations of
   * neurons in the hidden layers agrees well with the sparsity parameter 'rho'.
   *
   * @param parameters Current values of the model parameters.
   */
  double Evaluate(const arma::mat& parameters) const;

  /**
   * Evaluates the gradient values of the objective function given the current
   * set of parameters. The function performs a feedforward pass and computes
   * the error in reconstructing the data points. It then uses the
   * backpropagation algorithm to compute the gradient values.
   *
   * @param parameters Current values of the model parameters.
   * @param gradient Matrix where gradient values will be stored.
   */
  void Gradient(const arma::mat& parameters, arma::mat& gradient) const;

  /**
   * Returns the elementwise sigmoid of the passed matrix, where the sigmoid
   * function of a real number 'x' is [1 / (1 + exp(-x))].
   *
   * @param x Matrix of real values for which we require the sigmoid activation.
   */
  void Sigmoid(const arma::mat& x, arma::mat& output) const
  {
    output = (1.0 / (1 + arma::exp(-x)));
  }

  //! Return the initial point for the optimization.
  const arma::mat& GetInitialPoint() const { return initialPoint; }

  //! Sets size of the visible layer.
  void VisibleSize(const size_t visible)
  {
    this->visibleSize = visible;
  }

  //! Gets size of the visible layer.
  size_t VisibleSize() const
  {
    return visibleSize;
  }

  //! Sets size of the hidden layer.
  void HiddenSize(const size_t hidden)
  {
    this->hiddenSize = hidden;
  }

  //! Gets the size of the hidden layer.
  size_t HiddenSize() const
  {
    return hiddenSize;
  }

  //! Sets the L2-regularization parameter.
  void Lambda(const double l)
  {
    this->lambda = l;
  }

  //! Gets the L2-regularization parameter.
  double Lambda() const
  {
    return lambda;
  }

  //! Sets the KL divergence parameter.
  void Beta(const double b)
  {
    this->beta = b;
  }

  //! Gets the KL divergence parameter.
  double Beta() const
  {
    return beta;
  }

  //! Sets the sparsity parameter.
  void Rho(const double r)
  {
    this->rho = r;
  }

  //! Gets the sparsity parameter.
  double Rho() const
  {
    return rho;
  }

 private:
  //! The matrix of data points.
  const arma::mat& data;
  //! Intial parameter vector.
  arma::mat initialPoint;
  //! Size of the visible layer.
  size_t visibleSize;
  //! Size of the hidden layer.
  size_t hiddenSize;
  //! L2-regularization parameter.
  double lambda;
  //! KL divergence parameter.
  double beta;
  //! Sparsity parameter.
  double rho;
};

}; // namespace nn
}; // namespace mlpack

#endif
