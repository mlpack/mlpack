/**
 * @file methods/sparse_autoencoder/sparse_autoencoder_function_impl.hpp
 * @author Siddharth Agrawal
 *
 * Implementation of function to be optimized for sparse autoencoders.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_FUNCTION_IMPL_HPP

#include "sparse_autoencoder_function.hpp"

namespace mlpack {

inline SparseAutoencoderFunction::SparseAutoencoderFunction(
    const arma::mat& data,
    const size_t visibleSize,
    const size_t hiddenSize,
    const double lambda,
    const double beta,
    const double rho) :
    data(data),
    visibleSize(visibleSize),
    hiddenSize(hiddenSize),
    lambda(lambda),
    beta(beta),
    rho(rho)
{
  // Initialize the parameters to suitable values.
  initialPoint = InitializeWeights();
}

/** Initializes the parameter weights if the initial point is not passed to the
  * constructor. The weights w1, w2 are initialized to randomly in the range
  * [-r, r] where 'r' is decided using the sizes of the visible and hidden
  * layers. The biases b1, b2 are initialized to 0.
  */
inline const arma::mat SparseAutoencoderFunction::InitializeWeights()
{
  // The module uses a matrix to store the parameters, its structure looks like:
  //          vSize   1
  //       |        |  |
  //  hSize|   w1   |b1|
  //       |________|__|
  //       |        |  |
  //  hSize|   w2'  |  |
  //       |________|__|
  //      1|   b2'  |  |
  //
  // There are (hiddenSize + 1) empty cells in the matrix, but it is small
  // compared to the matrix size. The above structure allows for smooth matrix
  // operations without making the code too ugly.

  // Initialize w1 and w2 to random values in the range [0, 1], then set b1 and
  // b2 to 0.
  arma::mat parameters;
  parameters.randu(2 * hiddenSize + 1, visibleSize + 1);
  parameters.row(2 * hiddenSize).zeros();
  parameters.col(visibleSize).zeros();

  // Decide the parameter 'r' depending on the size of the visible and hidden
  // layers. The formula used is r = sqrt(6) / sqrt(vSize + hSize + 1).
  const double range = std::sqrt(6) / std::sqrt(visibleSize + hiddenSize + 1);

  // Shift range of w1 and w2 values from [0, 1] to [-r, r].
  parameters.submat(0, 0, 2 * hiddenSize - 1, visibleSize - 1) = 2 * range *
      (parameters.submat(0, 0, 2 * hiddenSize - 1, visibleSize - 1) - 0.5);

  return parameters;
}

/** Evaluates the objective function given the parameters.
  */
inline double SparseAutoencoderFunction::Evaluate(const arma::mat& parameters)
    const
{
  // The objective function is the average squared reconstruction error of the
  // network. w1 and b1 are the weights and biases associated with the hidden
  // layer, whereas w2 and b2 are associated with the output layer.
  // f(w1,w2,b1,b2) = sum((data - sigmoid(w2*sigmoid(w1data + b1) + b2))^2) / 2m
  // 'm' is the number of training examples.
  // The cost also takes into account the regularization and KL divergence terms
  // to control the parameter weights and sparsity of the model respectively.

  // Compute the limits for the parameters w1, w2, b1 and b2.
  const size_t l1 = hiddenSize;
  const size_t l2 = visibleSize;
  const size_t l3 = 2 * hiddenSize;

  // w1, w2, b1 and b2 are not extracted separately, 'parameters' is directly
  // used in their place to avoid copying data. The following representations
  // are used:
  // w1 <- parameters.submat(0, 0, l1-1, l2-1)
  // w2 <- parameters.submat(l1, 0, l3-1, l2-1).t()
  // b1 <- parameters.submat(0, l2, l1-1, l2)
  // b2 <- parameters.submat(l3, 0, l3, l2-1).t()

  arma::mat hiddenLayer, outputLayer;

  // Compute activations of the hidden and output layers.
  Sigmoid(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
      repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
      hiddenLayer);

  Sigmoid(parameters.submat(l1, 0, l3 - 1, l2 - 1).t() * hiddenLayer +
      repmat(parameters.submat(l3, 0, l3, l2 - 1).t(), 1, data.n_cols),
      outputLayer);

  arma::mat rhoCap, diff;

  // Average activations of the hidden layer.
  rhoCap = sum(hiddenLayer, 1) / data.n_cols;
  // Difference between the reconstructed data and the original data.
  diff = outputLayer - data;

  double wL2SquaredNorm;

  // Calculate squared L2-norms of w1 and w2.
  wL2SquaredNorm = accu(parameters.submat(0, 0, l3 - 1, l2 - 1) %
      parameters.submat(0, 0, l3 - 1, l2 - 1));

  double sumOfSquaresError, weightDecay, klDivergence, cost;

  // Calculate the reconstruction error, the regularization cost and the KL
  // divergence cost terms. 'sumOfSquaresError' is the average squared l2-norm
  // of the reconstructed data difference. 'weightDecay' is the squared l2-norm
  // of the weights w1 and w2. 'klDivergence' is the cost of the hidden layer
  // activations not being low. It is given by the following formula:
  // KL = sum_over_hSize(rho*log(rho/rhoCaq) + (1-rho)*log((1-rho)/(1-rhoCap)))
  sumOfSquaresError = 0.5 * accu(diff % diff) / data.n_cols;
  weightDecay = 0.5 * lambda * wL2SquaredNorm;
  klDivergence = beta * accu(rho * log(rho / rhoCap) + (1 - rho) *
      log((1 - rho) / (1 - rhoCap)));

  // The cost is the sum of the terms calculated above.
  cost = sumOfSquaresError + weightDecay + klDivergence;

  return cost;
}

/** Calculates and stores the gradient values given a set of parameters.
  */
inline void SparseAutoencoderFunction::Gradient(const arma::mat& parameters,
                                                arma::mat& gradient) const
{
  // Performs a feedforward pass of the neural network, and computes the
  // activations of the output layer as in the Evaluate() method. It uses the
  // Backpropagation algorithm to calculate the delta values at each layer,
  // except for the input layer. The delta values are then used with input layer
  // and hidden layer activations to get the parameter gradients.

  // Compute the limits for the parameters w1, w2, b1 and b2.
  const size_t l1 = hiddenSize;
  const size_t l2 = visibleSize;
  const size_t l3 = 2 * hiddenSize;

  // w1, w2, b1 and b2 are not extracted separately, 'parameters' is directly
  // used in their place to avoid copying data. The following representations
  // are used:
  // w1 <- parameters.submat(0, 0, l1-1, l2-1)
  // w2 <- parameters.submat(l1, 0, l3-1, l2-1).t()
  // b1 <- parameters.submat(0, l2, l1-1, l2)
  // b2 <- parameters.submat(l3, 0, l3, l2-1).t()

  arma::mat hiddenLayer, outputLayer;

  // Compute activations of the hidden and output layers.
  Sigmoid(parameters.submat(0, 0, l1 - 1, l2 - 1) * data +
      repmat(parameters.submat(0, l2, l1 - 1, l2), 1, data.n_cols),
      hiddenLayer);

  Sigmoid(parameters.submat(l1, 0, l3 - 1, l2 - 1).t() * hiddenLayer +
      repmat(parameters.submat(l3, 0, l3, l2 - 1).t(), 1, data.n_cols),
      outputLayer);

  arma::mat rhoCap, diff;

  // Average activations of the hidden layer.
  rhoCap = sum(hiddenLayer, 1) / data.n_cols;
  // Difference between the reconstructed data and the original data.
  diff = outputLayer - data;

  arma::mat klDivGrad, delOut, delHid;

  // The delta vector for the output layer is given by diff * f'(z), where z is
  // the preactivation and f is the activation function. The derivative of the
  // sigmoid function turns out to be f(z) * (1 - f(z)). For every other layer
  // in the neural network which comes before the output layer, the delta values
  // are given del_n = w_n' * del_(n+1) * f'(z_n). Since our cost function also
  // includes the KL divergence term, we adjust for that in the formula below.
  klDivGrad = beta * (-(rho / rhoCap) + (1 - rho) / (1 - rhoCap));
  delOut = diff % outputLayer % (1 - outputLayer);
  delHid = (parameters.submat(l1, 0, l3 - 1, l2 - 1) * delOut +
      repmat(klDivGrad, 1, data.n_cols)) % hiddenLayer %
      (1 - hiddenLayer);

  gradient.zeros(2 * hiddenSize + 1, visibleSize + 1);

  // Compute the gradient values using the activations and the delta values. The
  // formula also accounts for the regularization terms in the objective.
  // function.
  gradient.submat(0, 0, l1 - 1, l2 - 1) = delHid * data.t() / data.n_cols +
      lambda * parameters.submat(0, 0, l1 - 1, l2 - 1);
  gradient.submat(l1, 0, l3 - 1, l2 - 1) =
      (delOut * hiddenLayer.t() / data.n_cols +
      lambda * parameters.submat(l1, 0, l3 - 1, l2 - 1).t()).t();
  gradient.submat(0, l2, l1 - 1, l2) = sum(delHid, 1) / data.n_cols;
  gradient.submat(l3, 0, l3, l2 - 1) = (sum(delOut, 1) / data.n_cols).t();
}

} // namespace mlpack

#endif
