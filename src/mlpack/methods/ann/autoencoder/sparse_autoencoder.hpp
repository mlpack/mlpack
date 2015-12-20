/**
 * @file one_hot_layer.hpp
 * @author Shangtong Zhang
 *
 * Definition of the OneHotLayer class, which implements a standard network
 * layer.
 */
#ifndef __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_HPP
#define __MLPACK_METHODS_ANN_SPARSE_AUTOENCODER_HPP

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

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename HiddenActivate = BaseLayer<LogisticFunction>,
         typename OutputActivate = HiddenActivate,
         typename MatType = arma::mat,
         template<typename,typename> class Optimize = RMSPROP,
         typename HiddenLayer = SparseInputLayer
         <Optimize, RandomInitialization, MatType, MatType>,
         typename OutputLayer = SparseOutputLayer
         <Optimize, RandomInitialization, MatType, MatType>
         >
class SparseAutoencoder
{  
  using BiasLayer =
  SparseBiasLayer<Optimize, ZeroInitialization, MatType, MatType>;

  using Network = std::tuple<HiddenLayer, BiasLayer, HiddenActivate,
  OutputLayer, BiasLayer, OutputActivate>;

  using FFN = FFN<Network, OneHotLayer, SparseErrorFunction<MatType>>;
 public:
  /**
   * Construct sparse autoencoder function
   * @param visibleSize Visible size of the input data, it is same as the feature
   * size of the training data
   * @param hiddenSize Hidden size of the hidden layer, usually it should be
   * smaller than the visible size
   * @param batchSize The batch size used to train the network.
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoder(size_t visibleSize,
                    size_t hiddenSize,
                    size_t batchSize,
                    const double lambda = 0.0001,
                    const double beta = 3,
                    const double rho = 0.01) :
    range(std::sqrt(6) / std::sqrt(visibleSize + hiddenSize + 1)),
    encoder(std::make_tuple(
              HiddenLayer(visibleSize, hiddenSize, {-range, range},
                          lambda),
              BiasLayer(hiddenSize, batchSize),
              HiddenActivate(),
              OutputLayer(hiddenSize, visibleSize, {-range, range},
                          lambda, beta, rho),
              BiasLayer(visibleSize, batchSize),
              OutputActivate()),
            oneHotLayer,
            SparseErrorFunction<MatType>(lambda, beta, rho))
  {
  }

  /**
   * Train the sparse autoencoder network
   *
   * @param input Data used to train the network
   * @param maxEpochs The number of maximal trained iterations (0 means no
   * limit).
   * @param batchSize The batch size used to train the network.
   * @param tolerance Train the network until it converges against
   * the specified threshold.
   * @param shuffle If true, the order of the training set is shuffled;
   * otherwise, each data is visited in linear order.
   */
  void Train(MatType const &input,
             const size_t maxEpochs = 0,
             const double tolerance = 0.0001,
             const bool shuffle = true)
  {
     Trainer<FFN, MatType> trainer(encoder, maxEpochs,
                                   std::get<1>(encoder.Network()).BatchSize(),
                                   tolerance,
                                   shuffle);
     trainer.Train(input, input, input, input);
  }

  /**
   * Get the weights of decoder
   * @return Weights of decoder
   */
  MatType const& DecoderWeights() const
  {
    return std::get<3>(encoder.Network()).Weights();
  }

  MatType& DecoderWeights()
  {
    return std::get<3>(encoder.Network()).Weights();
  }

  /**
   * Get the bias of decoder
   * @return Bias of decoder
   */
  MatType const& DecoderBias() const
  {
    return std::get<4>(encoder.Network()).Weights();
  }  

  /**
   * Get the weights of encoder
   * @return Weights of encoder
   */
  MatType const& EncoderWeights() const
  {
    return std::get<0>(encoder.Network()).Weights();
  }

  MatType& EncoderWeights()
  {
    return std::get<0>(encoder.Network()).Weights();
  }

  /**
   * Get the bias of encoder
   * @return bias of encoder
   */
  MatType const& EncoderBias() const
  {
    return std::get<1>(encoder.Network()).Weights();
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
    std::get<2>(encoder.Network()).fn(encodedInput, output);
  }      

 private:  
  const double range;
  OneHotLayer oneHotLayer;
  FFN encoder;
};

template<typename MatType = arma::mat,
         template<typename,typename> class Optimize = RMSPROP>
using LogisticSparseAutoencoder = SparseAutoencoder<
    BaseLayer<LogisticFunction>,
    BaseLayer<LogisticFunction>,
    MatType,
    Optimize,
    SparseInputLayer<Optimize, RandomInitialization, MatType, MatType>,
    SparseOutputLayer<Optimize, RandomInitialization, MatType, MatType>>;

}; // namespace ann
}; // namespace mlpack

#endif
