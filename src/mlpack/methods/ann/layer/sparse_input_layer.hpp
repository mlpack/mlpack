/**
 * @file sparse_input_layer.hpp
 * @author Tham Ngap Wei
 *
 * Definition of the sparse input class which serve as the first layer
 * of the sparse autoencoder
 */
#ifndef __MLPACK_METHODS_ANN_LAYER_SPARSE_INPUT_LAYER_HPP
#define __MLPACK_METHODS_ANN_LAYER_SPARSE_INPUT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/optimizer/rmsprop.hpp>

#include <type_traits>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the SparseInputLayer. The SparseInputLayer class represents the
 * first layer of sparse autoencoder
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
    class WeightInitRule = RandomInitialization,
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
    >
class SparseInputLayer
{
 public:
  /**
   * Create the SparseInputLayer object using the specified number of units.
   *
   * @param inSize The number of input units.
   * @param outSize The number of output units.
   * @param WeightInitRule The weight initialization rule used to initialize the
   *        weight matrix.
   */
  SparseInputLayer(const size_t inSize,
                   const size_t outSize,                   
                   WeightInitRule weightInitRule = WeightInitRule(),
                   const double lambda = 0.0001) :
    inSize(inSize),
    outSize(outSize),
    lambda(lambda),
    optimizer(new OptimizerType<SparseInputLayer<OptimizerType,
                                                 WeightInitRule,
                                                 InputDataType,
                                                 OutputDataType>,
                                                 OutputDataType>(*this)),
    ownsOptimizer(true)
  {
    weightInitRule.Initialize(weights, outSize, inSize);
  }

  SparseInputLayer(SparseInputLayer &&layer) noexcept
  {
    *this = std::move(layer);
  }

  SparseInputLayer& operator=(SparseInputLayer &&layer) noexcept
  {
    ownsOptimizer = layer.ownsOptimizer;
    layer.ownsOptimizer = false;
    optimizer = layer.optimizer;
    layer.optimizer = nullptr;

    inSize = layer.inSize;
    outSize = layer.outSize;
    lambda = layer.lambda;
    weights.swap(layer.weights);
    delta.swap(layer.delta);
    gradient.swap(layer.gradient);
    inputParameter.swap(layer.inputParameter);
    outputParameter.swap(layer.outputParameter);

    return *this;
  }

  /**
   * Delete the linear layer object and its optimizer.
   */
  ~SparseInputLayer()
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
    output = weights * input;
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
                arma::Mat<eT>& g)
  {
    g = gy;
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
    g = d * inputParameter.t() /
        static_cast<typename InputDataType::value_type>(inputParameter.n_cols) +
        lambda * weights;
  }

  //! Get the optimizer.
  OptimizerType<SparseInputLayer<OptimizerType,
  WeightInitRule,
  InputDataType,
  OutputDataType>, OutputDataType>& Optimizer() const
  {
    return *optimizer;
  }
  //! Modify the optimizer.
  OptimizerType<SparseInputLayer<OptimizerType,
  WeightInitRule,
  InputDataType,
  OutputDataType>, OutputDataType>& Optimizer()
  {
    return *optimizer;
  }

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

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
  OutputDataType const& Gradient() const {return gradient; }
  //! Modify the gradient.
  OutputDataType& Gradient() { return gradient; }

  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    using mlpack::data::CreateNVP;

    ar & CreateNVP(inSize, "inSize");
    ar & CreateNVP(outSize, "outSize");
    ar & CreateNVP(lambda, "lambda");
    ar & CreateNVP(weights, "weights");
    ar & CreateNVP(delta, "delta");
    ar & CreateNVP(gradient, "gradient");
    ar & CreateNVP(inputParameter, "inputParameter");
    ar & CreateNVP(outputParameter, "outputParameter");
    ar & CreateNVP(optimizer, "optimizer");
    ar & CreateNVP(ownsOptimizer, "ownsOptimizer");

    optimizer->Function(*this);
  }

 private:
  /**
   * This constructor is design for boost::serialization
   */
  SparseInputLayer(){}

  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! L2-regularization parameter.
  double lambda;

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
  OptimizerType<SparseInputLayer<OptimizerType,
  WeightInitRule,
  InputDataType,
  OutputDataType>, OutputDataType>* optimizer;

  //! Parameter that indicates if the class owns a optimizer object.
  bool ownsOptimizer;
}; // class SparseInputLayer

//! Layer traits for the SparseInputLayer.
template<
    template<typename, typename> class OptimizerType,
    typename WeightInitRule,
    typename InputDataType,
    typename OutputDataType
    >
class LayerTraits<SparseInputLayer<
    OptimizerType, WeightInitRule, InputDataType, OutputDataType> >
{
public:
  static const bool IsBinary = false;
  static const bool IsOutputLayer = false;
  static const bool IsBiasLayer = false;
  static const bool IsLSTMLayer = false;
  static const bool IsConnection = true;
};

}; // namespace ann
}; // namespace mlpack

#endif
