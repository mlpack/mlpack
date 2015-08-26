/**
 * @file bias_layer.hpp
 * @author Marcus Edel
 *
 * Definition of the BiasLayer class.
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_BIAS_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_BIAS_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/nguyen_widrow_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * An implementation of a standard bias layer. The BiasLayer class represents a
 * single layer of a neural network.
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
    class WeightInitRule = NguyenWidrowInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class BiasLayer
{
 public:
  /**
   * Create the BiasLayer object using the specified number of units and bias
   * parameter.
   *
   * @param outSize The number of output units.
   * @param bias The bias value.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  BiasLayer(const size_t outSize,
            const double bias = 1,
            WeightInitRule weightInitRule = WeightInitRule()) :
      outSize(outSize),
      bias(bias),
      optimizer(new OptimizerType<BiasLayer<OptimizerType,
                                            WeightInitRule,
                                            InputDataType,
                                            OutputDataType>,
                                            InputDataType>(*this)),
      ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outSize, 1);
  }

  /**
   * Delete the bias layer object and its optimizer.
   */
  ~BiasLayer()
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
    output = input + (weights * bias);
  }

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
  {
    output = input;
    for (size_t s = 0; s < input.n_slices; s++)
    {
      output.slice(s) += weights(s) * bias;
    }
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
   * @param g The calculated gradient.
   * @param d The calculated error.
   */
  template<typename GradientDataType, typename eT>
  void Gradient(GradientDataType& g, arma::Cube<eT>& d)
  {
    g = arma::Mat<eT>(weights.n_rows, weights.n_cols);
    for (size_t s = 0; s < d.n_slices; s++)
    {
      g(s) = arma::accu(d.slice(s)) * bias;
    }
  }

  /*
   * Calculate the gradient using the output delta and the bias.
   *
   * @param g The calculated gradient.
   * @param d The calculated error.
   */
  template<typename GradientDataType, typename eT>
  void Gradient(GradientDataType& g, arma::Mat<eT>& d)
  {
    g = d * bias;
  }

  //! Get the optimizer.
  OptimizerType<BiasLayer<OptimizerType,
                          WeightInitRule,
                          InputDataType,
                          OutputDataType>, InputDataType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<BiasLayer<OptimizerType,
                          WeightInitRule,
                          InputDataType,
                          OutputDataType>, InputDataType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the weights.
  InputDataType& Weights() const { return weights; }
  //! Modify the weights.
  InputDataType& Weights() { return weights; }

  //! Get the input parameter.
  InputDataType& InputParameter() const {return inputParameter; }
  //! Modify the input parameter.
  InputDataType& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  OutputDataType& OutputParameter() const {return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the delta.
  OutputDataType& Delta() const {return delta; }
  //! Modify the delta.
  OutputDataType& Delta() { return delta; }

  //! Get the gradient.
  InputDataType& Gradient() const {return gradient; }
  //! Modify the gradient.
  InputDataType& Gradient() { return gradient; }

 private:
  //! Locally-stored number of output units.
  const size_t outSize;

  //! Locally-stored bias value.
  double bias;

  //! Locally-stored weight object.
  InputDataType weights;

  //! Locally-stored delta object.
  OutputDataType delta;

  //! Locally-stored delta object.
  InputDataType gradient;

  //! Locally-stored input parameter object.
  InputDataType inputParameter;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<BiasLayer<OptimizerType,
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
class LayerTraits<BiasLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = true;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

/**
 * Standard 2D-Bias-Layer.
 */
template <
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::cube
>
using BiasLayer2D = BiasLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType>;

}; // namespace ann
}; // namespace mlpack

#endif
