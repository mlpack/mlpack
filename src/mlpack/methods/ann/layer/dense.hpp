/**
 * @file dense.hpp
 * @author N Rajiv Vaidyanathan
 *
 * Definition of the Dense block class, which improves gradient and feature
 * propogation. Reduces the number of parameters.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DENSE_HPP
#define MLPACK_METHODS_ANN_LAYER_DENSE_HPP

#include "batch_norm.hpp"
#include "leaky_relu.hpp"
#include "convolution.hpp"
#include "dropout.hpp"

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The dense block is a layer which connects each layer
 * to every other layer in a feedforward fashion. Compared to
 * traditional neural networks with L connections (connecting
 * the subsequent layers), the dense block has L(L+1)/2 connections.
 * It alleviates the vanishing-gradient problem and strengthens
 * feature propagation. It also encourages feature reuse and
 * substantially reduces the number of parameters.
 * 
 * For more information, see the following.
 * 
 * @article{DBLP:journals/corr/HuangLW16a,
 * author    = {Gao Huang and
 *              Zhuang Liu and
 *              Kilian Q. Weinberger},
 * title     = {Densely Connected Convolutional Networks},
 * journal   = {CoRR},
 * volume    = {abs/1608.06993},
 * year      = {2016},
 * url       = {http://arxiv.org/abs/1608.06993},
 * archivePrefix = {arXiv},
 * eprint    = {1608.06993},
 * timestamp = {Mon, 10 Sep 2018 15:49:32 +0200},
 * biburl    = {https://dblp.org/rec/bib/journals/corr/HuangLW16a},
 * bibsource = {dblp computer science bibliography, https://dblp.org}
 * }
 */
template<typename InputDataType = arma::cube,
         typename OutputDataType = arma::cube>
class Dense
{
 public:
  //! Create the Dense object.
  Dense();

  /**
   * Create the Dense layer object for a specific configuration.
   * 
   * @param nb_layers The number of convolution blocks to append to the block.
   * @param growth_rate Growth rate of the dense block.
   * @param bottleneck Bottleneck convolution block is added to each convolution
   *        block if true.
   * @param dropout_rate Dropout rate for the dropout layer.
   * @param weight_decay Weight decay factor.
   */
  Dense(const size_t nb_layers, const size_t growth_rate,
    const bool bottleneck = false, const double dropout_rate = 0.2,
    const double weight_decay = 1e-4);

  /**
   * Create the Convolution block used for building dense blocks.
   * 
   * @param input Input data used for evaluating the specified function.
   * @param growth_rate Growth rate of the dense block.
   * @param bottleneck Bottleneck convolution block is added to each convolution
   *        block if true.
   * @param dropout_rate Dropout rate for the dropout layer.
   * @param weight_decay Weight decay factor.
   */
  template<typename eT>
  arma::cube conv_block(arma::Cube<eT>&& input, const size_t& input_size,
    const size_t& growth_rate, const bool& bottleneck,
    const double& dropout_rate, const double& weight_decay);

  /**
   * Ordinary feed forward pass of the dense layer.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>&& input, arma::Cube<eT>&& output);

  /**
   * Ordinary feed backward pass of the dense layer.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Cube<eT>&& /* input */,
                arma::Cube<eT>&& gy,
                arma::Cube<eT>&& g);

  /**
   * Calculate the gradient using the output delta and the input activations.
   *
   * @param input The input activations
   * @param error The calculated error
   * @param gradient The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Cube<eT>&& input,
                arma::Cube<eT>&& error,
                arma::Cube<eT>&& gradient);

  //! Get the input parameter.
  InputDataType const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const { return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType const& Gradient() const { return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  /**
   * Serialize the layer
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:

  std::vector<BatchNorm<>> bn_set, bn_set_back, bn_set_grad;

  std::vector<LeakyReLU<>> leaky_relu_set, leaky_relu_set_back;

  std::vector<Convolution<>> conv_set,
    conv_set_back, conv_set_grad;

  std::vector<Dropout<>> dropout_set, dropout_set_back;

  //! Locally-stored number of convolution blocks to append to the block.
  size_t nb_layers;

  //! Locally-stored growth rate of the dense block.
  size_t growth_rate;

  //! Locally-stored value to decide if bottleneck has to be added.
  bool bottleneck;

  //! Locally-stored dropout rate for the dropout layer.
  double dropout_rate;

  //! Locally-stored weight decay factor.
  double weight_decay;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;
}; // class Dense

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "dense_impl.hpp"

#endif
