/**
 * @file recurrent_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the RecurrentLayer class.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_RECURRENT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the RecurrentLayer class. Recurrent layers can be used
 * similarly to feed-forward layers except that the input isn't stored in the
 * inputParameter, instead it's in stored in the recurrentParameter.
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
    class WeightInitRule = NguyenWidrowInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class RecurrentLayer
{
 public:
  /**
   * Create the RecurrentLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  RecurrentLayer(const size_t inSize,
                 const size_t outSize,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inSize(outSize),
      outSize(outSize),
      optimizer(new OptimizerType<RecurrentLayer<OptimizerType,
                                                 WeightInitRule,
                                                 InputDataType,
                                                 OutputDataType>,
                                                 OutputDataType>(*this)),
      recurrentParameter(arma::zeros<InputDataType>(inSize, 1)),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outSize, inSize);
  }

  /**
   * Create the RecurrentLayer object using the specified number of units.
   *
   * @param outSize The number of output units.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  RecurrentLayer(const size_t outSize,
                 WeightInitRule weightInitRule = WeightInitRule()) :
      inSize(outSize),
      outSize(outSize),
      optimizer(new OptimizerType<RecurrentLayer<OptimizerType,
                                                 WeightInitRule,
                                                 InputDataType,
                                                 OutputDataType>,
                                                 OutputDataType>(*this)),
      recurrentParameter(arma::zeros<InputDataType>(outSize, 1)),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outSize, inSize);
  }

  RecurrentLayer(RecurrentLayer &&layer) noexcept
  {
    *this = std::move(layer);
  }

  RecurrentLayer& operator=(RecurrentLayer &&layer) noexcept
  {
    optimizer = layer.optimizer;
    ownsOptimizer = layer.ownsOptimizer;
    layer.optimizer = nullptr;
    layer.ownsOptimizer = false;

    inSize = layer.inSize;
    outSize = layer.outSize;
    weights.swap(layer.weights);
    delta.swap(layer.delta);
    gradient.swap(layer.gradient);
    inputParameter.swap(layer.inputParameter);
    outputParameter.swap(layer.outputParameter);
    recurrentParameter.swap(layer.recurrentParameter);

    return *this;
  }

  /**
   * Delete the RecurrentLayer object and its optimizer.
   */
  ~RecurrentLayer()
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
    output = input + weights * recurrentParameter;
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
  template<typename InputType, typename eT>
  void Backward(const InputType& /* unused */,
                const arma::Mat<eT>& gy,
                arma::mat& g)
  {
    g = (weights).t() * gy;
  }

  /*
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param d The calculated error.
   * @param g The calculated gradient.
   */
  template<typename eT, typename GradientDataType>
  void Gradient(const arma::Mat<eT>& d, GradientDataType& g)
  {
    g = d * recurrentParameter.t();
  }

  //! Get the optimizer.
  OptimizerType<RecurrentLayer<OptimizerType,
                               WeightInitRule,
                               InputDataType,
                               OutputDataType>,
                               OutputDataType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<RecurrentLayer<OptimizerType,
                               WeightInitRule,
                               InputDataType,
                               OutputDataType>, OutputDataType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the weights.
  OutputDataType& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the input parameter.
  InputDataType& RecurrentParameter() const {return recurrentParameter; }
  //! Modify the input parameter.
  InputDataType& RecurrentParameter() { return recurrentParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  OutputDataType& Gradient() const {return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

 private:
  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored weight object.
  OutputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored gradient object.
  OutputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<RecurrentLayer<OptimizerType,
                               WeightInitRule,
                               InputDataType,
                               OutputDataType>, OutputDataType>* optimizer;

  //! Locally-stored recurrent parameter object.
  InputDataType recurrentParameter;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;
}; // class RecurrentLayer

//! Layer traits for the recurrent layer.
template<
    template<typename, typename> class OptimizerType,
    typename WeightInitRule,
    typename InputDataType,
    typename OutputDataType
>
class LayerTraits<RecurrentLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

} // namespace ann
} // namespace mlpack

#endif
