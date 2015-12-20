/**
 * @file bias_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BiasLayer class.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_SPARSE_BIAS_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_SPARSE_BIAS_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/zero_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a bias layer design for sparse autoencoder.
 * The BiasLayer class represents a single layer of a neural network.
 *
 * A convenient typedef is given:
 *
 *  - 2DBiasLayer
 *
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = ZeroInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class SparseBiasLayer
{
 public:
  /**
   * Create the BiasLayer object using the specified number of units and bias
   * parameter.
   *
   * @param outSize The number of output units.
   * @param batchSize The batch size used to train the network.
   * @param bias The bias value.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  SparseBiasLayer(const size_t outSize,
                  const size_t batchSize,
                  WeightInitRule weightInitRule = WeightInitRule()) :
      outSize(outSize),
      batchSize(batchSize),
      optimizer(new OptimizerType<SparseBiasLayer<OptimizerType,
                                                  WeightInitRule,
                                                  InputDataType,
                                                  OutputDataType>,
                                                  InputDataType>(*this)),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outSize, 1);
  }
  
  SparseBiasLayer(SparseBiasLayer &&layer) noexcept
  {
    *this = std::move(layer);
  }

  SparseBiasLayer& operator=(SparseBiasLayer &&layer) noexcept
  {
    optimizer = layer.optimizer;    
    ownsOptimizer = layer.ownsOptimizer;
    layer.optimizer = nullptr;
    layer.ownsOptimizer = false;

    outSize = layer.outSize;   
    batchSize = layer.batchSize;
    weights.swap(layer.weights);
    delta.swap(layer.delta);
    gradient.swap(layer.gradient);
    inputParameter.swap(layer.inputParameter);
    outputParameter.swap(layer.outputParameter);

    return *this;
  }

  /**
   * Delete the bias layer object and its optimizer.
   */
  ~SparseBiasLayer()
  {
    if (ownsOptimizer)
      delete optimizer;
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output)
  {    
    output = input + arma::repmat(weights, 1, input.n_cols);
  }

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename DataType, typename ErrorType>
  void Backward(const DataType& /* unused */,
                const ErrorType& gy,
                ErrorType& g)
  {
    g = gy;
  }  

  /*
   * Calculate the gradient using the output delta and the bias.
   *
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Gradient(const arma::Mat<eT>& d, InputDataType& g)
  {
    using inputDataType = std::decay<decltype(inputParameter[0])>::type;
    g = arma::sum(d, 1) / static_cast<inputDataType>(batchSize);    
  }

  //! Get the optimizer.
  OptimizerType<SparseBiasLayer<OptimizerType,
                          WeightInitRule,
                          InputDataType,
                          OutputDataType>, InputDataType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<SparseBiasLayer<OptimizerType,
                          WeightInitRule,
                          InputDataType,
                          OutputDataType>, InputDataType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the batch size
  size_t BatchSize() const { return batchSize; }
  //! Modify the batch size
  size_t& BatchSize() { return batchSize; }

  //! Get the weights.
  InputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  InputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType const& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType const& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType const& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  InputDataType const& Gradient() const {return gradient; }
  //! Modify the gradient.
  InputDataType& Gradient() { return gradient; }

 private:
  //! Locally-stored number of output units.
  size_t outSize;

  //! The batch size used to train the network.
  size_t batchSize;

  //! Locally-stored weight object.
  InputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  InputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<SparseBiasLayer<OptimizerType,
                          WeightInitRule,
                          InputDataType,
                          OutputDataType>, InputDataType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;
}; // class BiasLayer

//! Layer traits for the bias layer.
template<
  template<typename, typename> class OptimizerType,
  typename WeightInitRule,
  typename InputDataType,
  typename OutputDataType
>
class LayerTraits<SparseBiasLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = true;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

} // namespace ann
} // namespace mlpack

#endif
