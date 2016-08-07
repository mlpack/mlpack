/**
 * @file concat_layer.hpp
 * @author Nilay Jain
 * @author Marcus Edel
 *
 * Definition of the ConcatLayer class.
 */

#ifndef MLPACK_METHODS_ANN_LAYER_CONCAT_LAYER_HPP
#define MLPACK_METHODS_ANN_LAYER_CONCAT_LAYER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/one_hot_layer.hpp>
#include <mlpack/methods/ann/layer/conv_layer.hpp>
#include <mlpack/methods/ann/layer/pooling_layer.hpp>
#include <mlpack/methods/ann/layer/softmax_layer.hpp>
#include <mlpack/methods/ann/layer/bias_layer.hpp>
#include <mlpack/methods/ann/layer/linear_layer.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {
/**
 * @tparam JoinLayers Contains all layer modules that need be concatenated.
 */
template <typename JoinLayers,
          typename InputDataType = arma::cube,
          typename OutputDataType = arma::cube>
class ConcatLayer
{
 public:
  ConcatLayer(const size_t numLayers,
              JoinLayers &&layers) :
  numLayers(numLayers),
  layers(std::forward<JoinLayers>(layers))
  {
    /* Nothing to do. */
  }

  template<typename eT>
  void Forward(const arma::Cube<eT>& , arma::Cube<eT>& output)
  {
    ForwardTail(layers, output);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& layers, DataType& output)
  {
    /* Nothing to do. */
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  ForwardTail(std::tuple<Tp...>& layers, DataType& output)
  {
    output = arma::join_slices(output, std::get<I>(layers).OutputParameter());
    ForwardTail<I + 1, DataType, Tp...>(layers, output);
  }

  template<typename eT>
  void Backward(arma::Cube<eT>&, arma::Cube<eT>& error, arma::Cube<eT>& )
  {
    size_t slice_idx = 0;
    BackwardTail(layers, error, slice_idx);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  BackwardTail(std::tuple<Tp...>& layers, const DataType& error,  size_t slice_idx)
  {
    /* Nothing to do. */
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  BackwardTail(std::tuple<Tp...>& layers, const DataType& error,  size_t slice_idx)
  {
    DataType subError = error.slices(slice_idx,
        slice_idx + std::get<I>(layers).OutputParameter().n_slices - 1);
    std::get<I>(layers).Backward(std::get<I>(layers).OutputParameter(), subError,
          std::get<I>(layers).Delta());
    slice_idx += std::get<I>(layers).OutputParameter().n_slices;
    BackwardTail<I + 1, DataType, Tp...>(layers, error, slice_idx);
  }

  template<typename eT>
  void Gradient(const arma::Cube<eT>&, arma::Cube<eT>& delta, arma::Cube<eT>&)
  {
    size_t slice_idx = 0;
    GradientTail(layers, delta, slice_idx);
  }

  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I == sizeof...(Tp), void>::type
  GradientTail(std::tuple<Tp...>& layers, const DataType& delta,  size_t slice_idx)
  { /* Nothing to do. */ }
  
  template<size_t I = 0, typename DataType, typename... Tp>
  typename std::enable_if<I < sizeof...(Tp), void>::type
  GradientTail(std::tuple<Tp...>& layers, const DataType& delta,  size_t slice_idx)
  {
    DataType deltaNext = delta.slices(slice_idx,
        slice_idx + std::get<I>(layers).OutputParameter().n_slices - 1);
    
    slice_idx = Update(layers, std::get<I>(layers).OutputParameter(), deltaNext);

    GradientTail<I + 1, DataType, Tp...>(layers, delta, slice_idx);
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      HasGradientCheck<T, P&(T::*)()>::value, size_t>::type
  Update(T& layer, P& /* unused */, D& delta, size_t slice_idx)
  {
    layer.Gradient(layer.InputParameter(), delta, layer.Gradient());
    slice_idx += layer.OutputParameter().n_slices;
    return slice_idx;
  }

  template<typename T, typename P, typename D>
  typename std::enable_if<
      !HasGradientCheck<T, P&(T::*)()>::value, size_t>::type
  Update(T& /* unused */, P& /* unused */, D& /* unused */, size_t slice_idx)
  {
    return slice_idx;
  }

 private:

  //! Get the weights.
  OutputDataType const& Weights() const { return weights; }
  //! Modify the weights.
  OutputDataType& Weights() { return weights; }

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

  //! number of layers to concatenate.
  size_t numLayers;

  //! Instantiated convolutional neural network.
  JoinLayers layers;

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
};

} // namespace ann
} // namspace mlpack

#endif
