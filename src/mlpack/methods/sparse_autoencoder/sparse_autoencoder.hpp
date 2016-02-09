/**
 * @file sparse_autoencoder.hpp
 * @author Siddharth Agrawal
 * @author Tham Ngap Wei
 *
 * Definition of the sparse autoencoder class, for automatic learning of
 * representative features.
 */
#ifndef __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP
#define __MLPACK_METHODS_SPARSE_AUTOENCODER_SPARSE_AUTOENCODER_HPP

#include <mlpack/core.hpp>

#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>

#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_bias_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_input_layer.hpp>
#include <mlpack/methods/ann/layer/sparse_output_layer.hpp>

#include <mlpack/methods/ann/optimizer/rmsprop.hpp>
#include <mlpack/methods/ann/trainer/trainer.hpp>

#include <mlpack/methods/ann/performance_functions/sparse_function.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

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
 * This implementation allows the use of arbitrary mlpack optimizers via the
 * Optimizer template parameter.
 *
 * @tparam MatType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam Optimizer The optimizer to use; by default this is RMSPROP.
 *
 * @tparam HiddenActivate Activation function used for the hidden layer.
 *
 * @tparam OutputActivate Activation function used for the output layer.
 *
 * @tparam HiddenLayer The layer type of the hidden layer, this type must
 *         provide functions "Forward(const InputType& input,
 *         OutputType& output)" and "Backward(const DataType& input,
 *         const DataType& gy, DataType& g)" the InputType, OutpuType, DataType
 *         must be able to accept arma::mat.
 *
 * @tparam OutputLayer The layer type of the output, this type must
 *         provide functions "Forward(const InputType& input,
 *         OutputType& output)" and "Backward(const DataType& input,
 *         const DataType& gy, DataType& g)" the InputType, OutpuType, DataType
 *         must be able to accept arma::mat.
 */
template<
    typename HiddenActivate = BaseLayer<LogisticFunction>,
    typename OutputActivate = HiddenActivate,
    typename MatType = arma::mat,
    template<typename> class Optimizer = RMSPROP,
    typename HiddenLayer = SparseInputLayer<
        Optimizer, RandomInitialization, MatType, MatType>,
    typename OutputLayer = SparseOutputLayer<
        Optimizer, RandomInitialization, MatType, MatType>
>
class SparseAutoencoder
{
  // Convenience typedefs for the internal model construction.
  using BiasLayer =
      SparseBiasLayer<Optimizer, ZeroInitialization, MatType, MatType>;

  using Network = std::tuple<HiddenLayer, BiasLayer, HiddenActivate,
      OutputLayer, BiasLayer, OutputActivate>;

  using FFNet = FFN<Network, OneHotLayer, SparseErrorFunction<MatType> >;

 public:
  /**
   * Construct sparse autoencoder function.
   *
   * @param visibleSize Visible size of the input data, it is same as the
   *        feature size of the training data.
   * @param hiddenSize Hidden size of the hidden layer, usually it should be
   *        smaller than the visible size.
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
        HiddenLayer(visibleSize, hiddenSize, {-range, range}, lambda),
        BiasLayer(hiddenSize, batchSize),
        HiddenActivate(),
        OutputLayer(hiddenSize, visibleSize, {-range, range}, lambda, beta, rho),
        BiasLayer(visibleSize, batchSize),
        OutputActivate()),
        oneHotLayer,
        SparseErrorFunction<MatType>(lambda, beta, rho))
  {
    // Nothing to do here.
  }

  /**
   * Construct sparse autoencoder function.
   *
   * @param data Input data with each column as one example.
   * @param visibleSize Visible size of the input data, it is same as the
   *        feature size of the training data.
   * @param hiddenSize Hidden size of the hidden layer, usually it should be
   *        smaller than the visible size.
   * @param batchSize The batch size used to train the network.
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseAutoencoder(const MatType& input,
                    size_t visibleSize,
                    size_t hiddenSize,
                    const double lambda = 0.0001,
                    const double beta = 3,
                    const double rho = 0.01) :
    range(std::sqrt(6) / std::sqrt(visibleSize + hiddenSize + 1)),
    encoder(std::make_tuple(
        HiddenLayer(visibleSize, hiddenSize, {-range, range}, lambda),
        BiasLayer(hiddenSize, input.n_rows),
        HiddenActivate(),
        OutputLayer(hiddenSize, visibleSize, {-range, range}, lambda, beta, rho),
        BiasLayer(visibleSize, input.n_rows),
        OutputActivate()),
        oneHotLayer,
        SparseErrorFunction<MatType>(lambda, beta, rho)),
    input(input)
  {
    // Nothing to do here.
  }

  /**
   * Train the sparse autoencoder network.
   *
   * @param input Data used to train the network
   * @param maxEpochs The number of maximal trained iterations (0 means no
   *        limit).
   * @param batchSize The batch size used to train the network.
   * @param tolerance Train the network until it converges against
   *        the specified threshold.
   * @param shuffle If true, the order of the training set is shuffled;
   *        otherwise, each data is visited in linear order.
   */
  void Train(MatType const &input,
             const size_t maxEpochs = 0,
             const double tolerance = 0.0001,
             const bool shuffle = true)
  {
     Trainer<FFNet, MatType> trainer(encoder, maxEpochs,
                                     std::get<1>(encoder.Model()).BatchSize(),
                                     tolerance,
                                     shuffle);
     trainer.Train(input, input, input, input);
  }

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
  double Evaluate(const arma::mat& parameters)
  {
    const size_t l1 = (parameters.n_rows - 1) / 2;
    const size_t l2 = parameters.n_cols - 1;
    const size_t l3 = 2 * l1;

    std::get<0>(encoder.Model()).Weights() = parameters.submat(
        0, 0, l1 - 1, l2 - 1);
    std::get<1>(encoder.Model()).Weights() = parameters.submat(
        0, l2, l1 - 1, l2);
    std::get<3>(encoder.Model()).Weights() = parameters.submat(
        l1, 0, l3 - 1, l2 - 1).t();
    std::get<4>(encoder.Model()).Weights() = parameters.submat(
        l3, 0, l3, l2 - 1).t();

    encoder.FeedForward(input, input, error);
    return encoder.Error();
  }

  /**
   * Get the weights of decoder.
   *
   * @return Weights of decoder
   */
  MatType const& DecoderWeights() const
  {
    return std::get<3>(encoder.Model()).Weights();
  }

  MatType& DecoderWeights()
  {
    return std::get<3>(encoder.Model()).Weights();
  }

  /**
   * Get the bias of decoder.
   *
   * @return Bias of decoder.
   */
  MatType const& DecoderBias() const
  {
    return std::get<4>(encoder.Model()).Weights();
  }  

  /**
   * Get the weights of encoder.
   *
   * @return Weights of encoder.
   */
  MatType const& EncoderWeights() const
  {
    return std::get<0>(encoder.Model()).Weights();
  }

  MatType& EncoderWeights()
  {
    return std::get<0>(encoder.Model()).Weights();
  }

  /**
   * Get the bias of encoder.
   *
   * @return bias of encoder.
   */
  MatType const& EncoderBias() const
  {
    return std::get<1>(encoder.Model()).Weights();
  }

  /**
   * Encode the input, in other words, extract the features of the input.
   *
   * @param input input data.
   * @param output features extracted from input.
   */
  void EncodeInput(MatType const &input, MatType &output) const
  {
    arma::mat const encodedInput = EncoderWeights() * input + arma::repmat(
        EncoderBias(), 1, input.n_cols);
    std::get<2>(encoder.Model()).fn(encodedInput, output);
  }

  //! Get the constructed model object.
  FFNet const& Model() const { return encoder; }
  //! Modify the constructed model object.
  FFNet& Model() { return encoder; }

 private:
  //! Locally-stored parameter that specifies the weight initialization range.
  const double range;

  //! Locally-stored output layer.
  OneHotLayer oneHotLayer;

  //! Locally-stored sparse autoencoder object.
  FFNet encoder;

  //! Locally-stored error parameter.
  MatType error;

  //! Locally-stored input parameter.
  MatType input;
};

// Convenience typedefs.

/**
 * Standard sparse autoencoder using the logistic activation function.
 */
template<
    typename MatType = arma::mat,
    template<typename> class Optimizer = RMSPROP
>
using LogisticSparseAutoencoder = SparseAutoencoder<
    BaseLayer<LogisticFunction>,
    BaseLayer<LogisticFunction>,
    MatType,
    Optimizer,
    SparseInputLayer<Optimizer, RandomInitialization, MatType, MatType>,
    SparseOutputLayer<Optimizer, RandomInitialization, MatType, MatType> >;

} // namespace ann
} // namespace mlpack

#endif
