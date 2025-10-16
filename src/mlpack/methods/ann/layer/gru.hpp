/**
 * @file methods/ann/layer/gru.hpp
 * @author Sumedh Ghaisas
 * @author Zachary Ng
 *
 * Definition of the GRU layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_GRU_HPP
#define MLPACK_METHODS_ANN_LAYER_GRU_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * An implementation of a gru network layer using the following algorithm.
 *
 * r_t = sigmoid(W_r x_t + U_r y_{t - 1})
 * z_t = sigmoid(W_z x_t + U_z y_{t - 1})
 * h_t =    tanh(W_h x_t + r_t % (U_h y_{t - 1}))
 * y_t =        (1 - z_t) % y_{t - 1} + z_t % h_t
 *
 * For more information, read the following paper:
 *
 * @code
 * @inproceedings{chung2015gated,
 *    title     = {Gated Feedback Recurrent Neural Networks},
 *    author    = {Chung, Junyoung and G{\"u}l{\c{c}}ehre, Caglar and Cho,
 *                Kyunghyun and Bengio, Yoshua},
 *    booktitle = {ICML},
 *    pages     = {2067--2075},
 *    year      = {2015},
 *    url       = {https://arxiv.org/abs/1502.02367}
 * }
 * @endcode
 *
 * This cell can be used in RNNs.
 *
 * @tparam MatType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <typename MatType = arma::mat>
class GRU : public RecurrentLayer<MatType>
{
 public:
  // Create the GRU object.
  GRU();

  /**
   * Create the GRU layer object using the specified parameters.
   *
   * @param outSize The number of output units.
   */
  GRU(const size_t outSize);

  // Clone the GRU object. This handles polymorphism correctly.
  GRU* Clone() const { return new GRU(*this); }

  // Copy the given GRU object.
  GRU(const GRU& other);
  // Take ownership of the given GRU object's data.
  GRU(GRU&& other);
  // Copy the given GRU object.
  GRU& operator=(const GRU& other);
  // Take ownership of the given GRU object's data.
  GRU& operator=(GRU&& other);

  virtual ~GRU() { }

  /**
   * Reset the layer parameter. The method is called to
   * assign the allocated memory to the internal learnable parameters.
   */
  void SetWeights(const MatType& weightsIn);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  void Forward(const MatType& input, MatType& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The input data (x) given to the forward pass.
   * @param output The propagated data (f(x)) resulting from Forward()
   * @param gy Propagated error from next layer.
   * @param g Matrix to store propagated error in for previous layer.
   */
  void Backward(const MatType& /* input */,
                const MatType& output,
                const MatType& gy,
                MatType& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input Original input data provided to Forward().
   * @param error Error as computed by `Backward()`.
   * @param gradient Matrix to store the gradients in.
   */
  void Gradient(const MatType& input,
                const MatType& /* error */,
                MatType& gradient);

  // Get the parameters.
  MatType const& Parameters() const { return weights; }
  // Modify the parameters.
  MatType& Parameters() { return weights; }

  // Get the total number of trainable parameters.
  size_t WeightSize() const;

  // Get the total number of recurrent state parameters.
  size_t RecurrentSize() const;

  // Given a properly set InputDimensions(), compute the output dimensions.
  void ComputeOutputDimensions()
  {
    inSize = this->inputDimensions[0];
    for (size_t i = 1; i < this->inputDimensions.size(); ++i)
      inSize *= this->inputDimensions[i];
    this->outputDimensions = std::vector<size_t>(this->inputDimensions.size(),
        1);

    // The GRU layer flattens its input.
    this->outputDimensions[0] = outSize;
  }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  // Locally-stored number of input units.
  size_t inSize;

  // Locally-stored number of output units.
  size_t outSize;

  // Locally-stored weight object.
  MatType weights;

  // Weight aliases for input connections.
  MatType resetGateWeight;
  MatType updateGateWeight;
  MatType hiddenGateWeight;

  // Weight aliases for recurrent connections.
  MatType recurrentResetGateWeight;
  MatType recurrentUpdateGateWeight;
  MatType recurrentHiddenGateWeight;

  // Recurrent state aliases.
  MatType resetGate;
  MatType updateGate;
  MatType hiddenGate;
  MatType currentOutput;
  MatType prevOutput;

  // Backwards workspace
  MatType deltaReset;
  MatType deltaUpdate;
  MatType deltaHidden;

  // Backwards workspace of previous step
  MatType nextDeltaReset;
  MatType nextDeltaUpdate;
  MatType nextDeltaHidden;

  // Makes internal state aliases from the recurrent state.
  void MakeStateAliases(size_t batchSize);
}; // class GRU

} // namespace mlpack

// Include implementation.
#include "gru_impl.hpp"

#endif
