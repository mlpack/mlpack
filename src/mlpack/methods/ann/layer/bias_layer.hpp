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
 * @tparam OptimizerType Type of the optimizer used to update the weights.
 * @tparam WeightInitRule Rule used to initialize the weight matrix.
 * @tparam DataType Type of data (arma::colvec, arma::mat arma::sp_mat or
 * arma::cube).
 */
template <
    template<typename, typename> class OptimizerType = mlpack::ann::RMSPROP,
    class WeightInitRule = NguyenWidrowInitialization,
    typename DataType = arma::mat
>
class BiasLayer
{
 public:
  /**
   * Create the BiasLayer object using the specified number of units and bias
   * parameter.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param bias The bias value.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  BiasLayer(const size_t inSize,
            const size_t outSize,
            const double bias = 1,
            WeightInitRule weightInitRule = WeightInitRule()) :
      inSize(inSize),
      outSize(outSize),
      bias(bias),
      optimizer(new OptimizerType<BiasLayer<OptimizerType,
                                              WeightInitRule,
                                              DataType>, DataType>(*this)),
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
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param input The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* unused */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g)
  {
    g = gy;
  }

  /*
   * Calculate the gradient using the output delta and the bias.
   *
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Gradient(arma::Mat<eT>& gradient)
  {
    gradient = delta * bias;
  }

  //! Get the optimizer.
  OptimizerType<BiasLayer<OptimizerType,
                          WeightInitRule,
                          DataType>, DataType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<BiasLayer<OptimizerType,
                          WeightInitRule,
                          DataType>, DataType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the weights.
  DataType& Weights() const { return weights; }
  //! Modify the weights.
  DataType& Weights() { return weights; }

  //! Get the parameter.
  DataType& Parameter() const {return parameter; }
  //! Modify the parameter.
  DataType& Parameter() { return parameter; }

  //! Get the delta.
  DataType& Delta() const {return delta; }
  //! Modify the delta.
  DataType& Delta() { return delta; }

 private:
  //! Locally-stored number of input units.
  const size_t inSize;

  //! Locally-stored number of output units.
  const size_t outSize;

  //! Locally-stored bias value.
  double bias;

  //! Locally-stored weight object.
  DataType weights;

  //! Locally-stored delta object.
  DataType delta;

  //! Locally-stored parameter object.
  DataType parameter;

  //! Locally-stored pointer to the optimzer object.
  OptimizerType<BiasLayer<OptimizerType,
                          WeightInitRule,
                          DataType>, DataType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;
}; // class BiasLayer

//! Layer traits for the bias layer.
template<
  template<typename, typename> class OptimizerType,
  typename WeightInitRule,
  typename DataType
>
class LayerTraits<BiasLayer<OptimizerType, WeightInitRule, DataType> >
{
 public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = true;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

}; // namespace ann
}; // namespace mlpack

#endif
