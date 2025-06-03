// Temporarily drop.
/**
 * @file methods/ann/layer/gru.hpp
 * @author Sumedh Ghaisas
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

#include <list>
#include <limits>

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * An implementation of a gru network layer.
 *
 * For more information, read the following paper:
 *
 * @code
 * @inproceedings{chung2015gated,
 *    title     = {Gated Feedback Recurrent Neural Networks.},
 *    author    = {Chung, Junyoung and G{\"u}l{\c{c}}ehre, Caglar and Cho,
                  Kyunghyun and Bengio, Yoshua},
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
class GRUType : public RecurrentLayer<MatType>
{
 public:
  // Create the GRU object.
  GRUType();

  /**
   * Create the GRU layer object using the specified parameters.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param rho Maximum number of steps to backpropagate through time (BPTT).
   */
  GRUType(const size_t inSize,
          const size_t outSize,
          const size_t rho = std::numeric_limits<size_t>::max());

  //! Clone the GRUType object. This handles polymorphism correctly.
  GRUType* Clone() const { return new GRUType(*this); }

  //! Copy the given GRUType object.
  GRUType(const GRUType& other);
  //! Take ownership of the given GRUType object's data.
  GRUType(GRUType&& other);
  //! Copy the given GRUType object.
  GRUType& operator=(const GRUType& other);
  //! Take ownership of the given GRUType object's data.
  GRUType& operator=(GRUType&& other);

  virtual ~GRUType() { }

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
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  void Backward(const MatType& /* input */,
                const MatType& gy,
                MatType& g);

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param input The input parameter used for calculating the gradient.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  void Gradient(const MatType& input,
                const MatType& /* error */,
                MatType& /* gradient */);

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

  // Locally-stored input 2 gate module.
  RecurrentLayer<MatType>* input2GateModule;

  // Locally-stored output 2 gate module.
  RecurrentLayer<MatType>* output2GateModule;

  // Locally-stored output hidden state 2 gate module.
  RecurrentLayer<MatType>* outputHidden2GateModule;

  // Locally-stored input gate module.
  RecurrentLayer<MatType>* inputGateModule;

  // Locally-stored hidden state module.
  RecurrentLayer<MatType>* hiddenStateModule;

  // Locally-stored forget gate module.
  RecurrentLayer<MatType>* forgetGateModule;
}; // class GRU

// Standard GRU layer
using GRU = GRUType<arma::mat>;

} // namespace mlpack

// Include implementation.
#include "gru_impl.hpp"

#endif
