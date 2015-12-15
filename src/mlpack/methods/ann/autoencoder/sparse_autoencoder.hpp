/**
 * @file sparse_autoencoder.hpp
 * @author Tham Ngap Wei
 *
 * Definition of the SparseAutoencoderFunction class, which implements a sparse autoencoder
 */
#ifndef __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_FUNCTION_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_bias_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_input_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_output_layer.hpp>

#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

#include <mlpack/methods/ann/performance_functions/sparse_function.hpp>

#include <memory>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename HiddenActivate = BaseLayer<LogisticFunction>,
         typename OutputActivate = HiddenActivate,
         typename Optimize = RMSPROP,
         typename MatType = arma::mat>
class SparseAutoencoderFunction
{
  using HiddenLayer =
  SparseInputLayer<Optimize, RandomInitialization, MatType, MatType>;
  using BiasLayer =
  SparseBiasLayer<Optimize, ZeroInitialization, MatType, MatType>;
  using OutputLayer =
  SparseOutputLayer<Optimize, RandomInitialization, MatType, MatType>;

  using Network = std::tuple<HiddenLayer, BiasLayer, HiddenActivate,
  OutputLayer, BiasLayer, OutputActivate>;

  using FFN = FFN<Network, OneHotLayer>;
 public:
  /**
   * Construct sparse autoencoder function
   * @param visibleSize Visible size of the input data, it is same as the feature
   * size of the training data
   * @param hiddenSize Hidden size of the hidden layer, usually it should be
   * smaller than the visible size
   * @param sampleSize The size of the training example
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoderFunction(size_t visibleSize,
                            size_t hiddenSize,
                            size_t sampleSize,
                            const double lambda = 0.0001,
                            const double beta = 3,
                            const double rho = 0.01) :
    range(std::sqrt(6) / std::sqrt(visibleSize + hiddenSize + 1)),
    encoder(std::make_tuple(
              HiddenLayer(visibleSize, hiddenSize, {range, range},
                          lambda),
              BiasLayer(hiddenSize, sampleSize),
              HiddenActivate(),
              OutputLayer(hiddenSize, visibleSize, {-range, range},
                          lambda, beta, rho),
              BiasLayer(visibleSize, sampleSize),
              OutputActivate()),
            oneHotLayer,
            SparseErrorFunction<MatType>(lambda, beta, rho))
  {
  }

  /**
   * Get the weights of decoder
   * @return Weights of decoder
   */
  MatType const& DecoderWeights() const
  {
    return std::get<3>(FFN.Network()).Weights();
  }

  /**
   * Get the bias of decoder
   * @return Bias of decoder
   */
  MatType const& DecoderBias() const
  {
    return std::get<4>(FFN.Network()).Weights();
  }

  /**
   * Get the weights of encoder
   * @return Weights of encoder
   */
  MatType const& EncoderWeights() const
  {
    return std::get<0>(FFN.Network()).Weights();
  }

  /**
   * Get the bias of encoder
   * @return bias of encoder
   */
  MatType const& EncoderBias() const
  {
    return std::get<1>(FFN.Network()).Weights();
  }

  /**
   * Encode the input, in other words, extract the features
   * of the input
   * @param input input data
   * @param output features extracted from input
   */
  void EncodeInput(MatType const &input, MatType &output) const
  {
    arma::mat const encodedInput = EncoderWeights() * input +
                                   arma::repmat(EncoderBias(),
                                                1, input.n_cols);
    HiddenActivate::fn(encodedInput, output);
  }

  /**
   * Run a single iteration of the feed forward algorithm, using the given
   * input and target vector, store the calculated error into the error
   * parameter.
   *
   * @param input Input data used to evaluate the network.
   * @param target Target data used to calculate the network error.
   * @param error The calulated error of the output layer.
   */
  template <typename InputType, typename TargetType, typename ErrorType>
  void FeedForward(const InputType& input,
                   const TargetType& target,
                   ErrorType& error)
  {
    encoder.FeedForward(input, target, error);
  }

  /**
   * Run a single iteration of the feed backward algorithm, using the given
   * error of the output layer.
   *
   * @param error The calulated error of the output layer.
   */
  template <typename InputType, typename ErrorType>
  void FeedBackward(const InputType& /* unused */, const ErrorType& error)
  {
    encoder.FeedBackward(InputType(), error);
  }

  /**
   * Update the weights using the layer defined optimizer.
   */
  void ApplyGradients()
  {
    encoder.ApplyGradients();
  }

  /**
   * Evaluate the trained network using the given input and compare the output
   * with the given target vector.
   *
   * @param input Input data used to evaluate the trained network.
   * @param target Target data used to calculate the network error.
   * @param error The calulated error of the output layer.
   */
  template <typename InputType, typename TargetType, typename ErrorType>
  double Evaluate(const InputType& input,
                  const TargetType& target,
                  ErrorType& error)
  {
    return encoder.Evaluate(input, target, error);
  }

  //! Get the error of the network.
  double Error() const
  {
    return encoder.Error();
  }

 private:  
  const double range;
  OneHotLayer oneHotLayer;
  FFN encoder;
};

}; // namespace ann
}; // namespace mlpack
