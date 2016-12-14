/**
 * @file sparse_autoencoder.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of sparse autoencoders.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP
#define MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "sparse_autoencoder_function.hpp"

namespace mlpack {
namespace nn {

/**
 * A sparse autoencoder is a neural network whose aim to learn compressed
 * representations of the data, typically for dimensionality reduction, with a
 * constraint on the activity of the neurons in the network. Sparse autoencoders
 * can be stacked together to learn a hierarchy of features, which provide a
 * better representation of the data for classification. This is a method used
 * in the recently developed field of deep learning. More technical details
 * about the model can be found on the following webpage:
 *
 * http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Data matrix.
 * const size_t vSize = 64; // Size of visible layer, depends on the data.
 * const size_t hSize = 25; // Size of hidden layer, depends on requirements.
 *
 * // Train the model using default options.
 * SparseAutoencoder encoder1(data, vSize, hSize);
 *
 * const size_t numBasis = 5; // Parameter required for L-BFGS algorithm.
 * const size_t numIterations = 100; // Maximum number of iterations.
 *
 * // Use an instantiated optimizer for the training.
 * SparseAutoencoderFunction saf(data, vSize, hSize);
 * L_BFGS<SparseAutoencoderFunction> optimizer(saf, numBasis, numIterations);
 * SparseAutoencoder<L_BFGS> encoder2(optimizer);
 *
 * arma::mat features1, features2; // Matrices for storing new representations.
 *
 * // Get new representations from the trained models.
 * encoder1.GetNewFeatures(data, features1);
 * encoder2.GetNewFeatures(data, features2);
 * @endcode
 *
 * This implementation allows the use of arbitrary mlpack optimizers via the
 * OptimizerType template parameter.
 *
 * @tparam OptimizerType The optimizer to use; by default this is L-BFGS.  Any
 *     mlpack optimizer can be used here.
 */
template<
  template<typename> class OptimizerType = mlpack::optimization::L_BFGS
>
class SparseAutoencoder
{
 public:
  /**
   * Construct the sparse autoencoder model with the given training data. This
   * will train the model. The parameters 'lambda', 'beta' and 'rho' can be set
   * optionally. Changing these parameters will have an effect on regularization
   * and sparsity of the model.
   *
   * @param data Input data with each column as one example.
   * @param visibleSize Size of input vector expected at the visible layer.
   * @param hiddenSize Size of input vector expected at the hidden layer.
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoder(const arma::mat& data,
                    const size_t visibleSize,
                    const size_t hiddenSize,
                    const double lambda = 0.0001,
                    const double beta = 3,
                    const double rho = 0.01);

  /**
   * Construct the sparse autoencoder model with the given training data. This
   * will train the model. This overload takes an already instantiated optimizer
   * and uses it to train the model. The optimizer should hold an instantiated
   * SparseAutoencoderFunction object for the function to operate upon. This
   * option should be preferred when the optimizer options are to be changed.
   *
   * @param optimizer Instantiated optimizer with instantiated error function.
   */
  SparseAutoencoder(OptimizerType<SparseAutoencoderFunction>& optimizer);

  /**
   * Transforms the provided data into the representation learned by the sparse
   * autoencoder. The function basically performs a feedforward computation
   * using the learned weights, and returns the hidden layer activations.
   *
   * @param data Matrix of the provided data.
   * @param features The hidden layer representation of the provided data.
   */
  void GetNewFeatures(arma::mat& data, arma::mat& features);

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
  //! Parameters after optimization.
  arma::mat parameters;
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

} // namespace nn
} // namespace mlpack

// Include implementation.
#include "sparse_autoencoder_impl.hpp"

#endif
