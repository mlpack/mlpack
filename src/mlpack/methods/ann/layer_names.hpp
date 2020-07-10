/**
 * @file methods/ann/layer_names.hpp
 * @author Sreenik Seal
 *
 * Implementation of a class that converts a given ann layer to string format.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant/static_visitor.hpp>
#include <string>

using namespace mlpack::ann;

/**
 * Implementation of a class that returns the string representation of the
 * name of the given layer.
 */
class LayerNameVisitor : public boost::static_visitor<std::string>
{
 public:
  //! Create the LayerNameVisitor object.
  LayerNameVisitor()
  {
  }

  /**
   * Return the name of the given layer of type AdaptiveMaxPooling as string.
   *
   * @param * Given layer of type AdaptiveMaxPooling.
   * @return The string representation of the layer.
   */
  std::string LayerString(AdaptiveMaxPooling<> * /*layer*/) const
  {
    return "adaptivemaxpooling";
  }

  /**
   * Return the name of the given layer of type AdaptiveMeanPooling as string.
   *
   * @param * Given layer of type AdaptiveMeanPooling.
   * @return The string representation of the layer.
   */
  std::string LayerString(AdaptiveMeanPooling<> * /*layer*/) const
  {
    return "adaptivemeanpooling";
  }

  /**
   * Return the name of the given layer of type AtrousConvolution as a string.
   *
   * @param * Given layer of type AtrousConvolution.
   * @return The string representation of the layer.
   */
  std::string LayerString(AtrousConvolution<>* /*layer*/) const
  {
    return "atrousconvolution";
  }

  /**
   * Return the name of the given layer of type AlphaDropout as a string.
   *
   * @param * Given layer of type AlphaDropout.
   * @return The string representation of the layer.
   */
  std::string LayerString(AlphaDropout<>* /*layer*/) const
  {
    return "alphadropout";
  }

  /**
   * Return the name of the given layer of type BatchNorm as a string.
   *
   * @param * Given layer of type BatchNorm.
   * @return The string representation of the layer.
   */
  std::string LayerString(BatchNorm<>* /*layer*/) const
  {
    return "batchnorm";
  }

  /**
   * Return the name of the given layer of type Constant as a string.
   *
   * @param * Given layer of type Constant.
   * @return The string representation of the layer.
   */
  std::string LayerString(Constant<>* /*layer*/) const
  {
    return "constant";
  }

  /**
   * Return the name of the given layer of type Convolution as a string.
   *
   * @param * Given layer of type Convolution.
   * @return The string representation of the layer.
   */
  std::string LayerString(Convolution<>* /*layer*/) const
  {
    return "convolution";
  }

  /**
   * Return the name of the given layer of type DropConnect as a string.
   *
   * @param * Given layer of type DropConnect.
   * @return The string representation of the layer.
   */
  std::string LayerString(DropConnect<>* /*layer*/) const
  {
    return "dropconnect";
  }

  /**
   * Return the name of the given layer of type Dropout as a string.
   *
   * @param * Given layer of type Dropout.
   * @return The string representation of the layer.
   */
  std::string LayerString(Dropout<>* /*layer*/) const
  {
    return "dropout";
  }

  /**
   * Return the name of the given layer of type FlexibleReLU as a string.
   *
   * @param * Given layer of type FlexibleReLU.
   * @return The string representation of the layer.
   */
  std::string LayerString(FlexibleReLU<>* /*layer*/) const
  {
    return "flexiblerelu";
  }

  /**
   * Return the name of the given layer of type LayerNorm as a string.
   *
   * @param * Given layer of type LayerNorm.
   * @return The string representation of the layer.
   */
  std::string LayerString(LayerNorm<>* /*layer*/) const
  {
    return "layernorm";
  }

  /**
   * Return the name of the given layer of type Linear as a string.
   *
   * @param * Given layer of type Linear.
   * @return The string representation of the layer.
   */
  std::string LayerString(Linear<>* /*layer*/) const
  {
    return "linear";
  }

  /**
   * Return the name of the given layer of type LinearNoBias as a string.
   *
   * @param * Given layer of type LinearNoBias.
   * @return The string representation of the layer.
   */
  std::string LayerString(LinearNoBias<>* /*layer*/) const
  {
    return "linearnobias";
  }

  /**
   * Return the name of the given layer of type NoisyLinear as a string.
   *
   * @param * Given layer of type NoisyLinear.
   * @return The string representation of the layer.
   */
  std::string LayerString(NoisyLinear<>* /*layer*/) const
  {
    return "noisylinear";
  }

  /**
   * Return the name of the given layer of type MaxPooling as a string.
   *
   * @param * Given layer of type MaxPooling.
   * @return The string representation of the layer.
   */
  std::string LayerString(MaxPooling<>* /*layer*/) const
  {
    return "maxpooling";
  }

  /**
   * Return the name of the given layer of type MeanPooling as a string.
   *
   * @param * Given layer of type MeanPooling.
   * @return The string representation of the layer.
   */
  std::string LayerString(MeanPooling<>* /*layer*/) const
  {
    return "meanpooling";
  }

  /**
   * Return the name of the given layer of type MultiplyConstant as a string.
   *
   * @param * Given layer of type MultiplyConstant.
   * @return The string representation of the layer.
   */
  std::string LayerString(MultiplyConstant<>* /*layer*/) const
  {
    return "multiplyconstant";
  }

  /**
   * Return the name of the given layer of type ReLULayer as a string.
   *
   * @param * Given layer of type ReLULayer.
   * @return The string representation of the layer.
   */
  std::string LayerString(ReLULayer<>* /*layer*/) const
  {
    return "relu";
  }

  /**
   * Return the name of the given layer of type TransposedConvolution as a
   * string.
   *
   * @param * Given layer of type TransposedConvolution.
   * @return The string representation of the layer.
   */
  std::string LayerString(TransposedConvolution<>* /*layer*/) const
  {
    return "transposedconvolution";
  }

  /**
   * Return the name of the given layer of type IdentityLayer as a string.
   *
   * @param * Given layer of type IdentityLayer.
   * @return The string representation of the layer.
   */
  std::string LayerString(IdentityLayer<>* /*layer*/) const
  {
    return "identity";
  }

  /**
   * Return the name of the given layer of type TanHLayer as a string.
   *
   * @param * Given layer of type TanHLayer.
   * @return The string representation of the layer.
   */
  std::string LayerString(TanHLayer<>* /*layer*/) const
  {
    return "tanh";
  }

  /**
   * Return the name of the given layer of type ELU as a string.
   *
   * @param * Given layer of type ELU.
   * @return The string representation of the layer.
   */
  std::string LayerString(ELU<>* /*layer*/) const
  {
    return "elu";
  }

  /**
   * Return the name of the given layer of type HardTanH as a string.
   *
   * @param * Given layer of type HardTanH.
   * @return The string representation of the layer.
   */
  std::string LayerString(HardTanH<>* /*layer*/) const
  {
    return "hardtanh";
  }

  /**
   * Return the name of the given layer of type LeakyReLU as a string.
   *
   * @param * Given layer of type LeakyReLU.
   * @return The string representation of the layer.
   */
  std::string LayerString(LeakyReLU<>* /*layer*/) const
  {
    return "leakyrelu";
  }

  /**
   * Return the name of the given layer of type PReLU as a string.
   *
   * @param * Given layer of type PReLU.
   * @return The string representation of the layer.
   */
  std::string LayerString(PReLU<>* /*layer*/) const
  {
    return "prelu";
  }

  /**
   * Return the name of the given layer of type SigmoidLayer as a string.
   *
   * @param * Given layer of type SigmoidLayer.
   * @return The string representation of the layer.
   */
  std::string LayerString(SigmoidLayer<>* /*layer*/) const
  {
    return "sigmoid";
  }

  /**
   * Return the name of the given layer of type LogSoftMax as a string.
   *
   * @param * Given layer of type LogSoftMax.
   * @return The string representation of the layer.
   */
  std::string LayerString(LogSoftMax<>* /*layer*/) const
  {
    return "logsoftmax";
  }

  /*
   * Return the name of the given layer of type LSTM as a string.
   *
   * @param * Given layer of type LSTM.
   * @return The string representation of the layer.
   */
  std::string LayerString(LSTM<>* /*layer*/) const
  {
    return "lstm";
  }

  /**
   * Return the name of the given layer of type CReLU as a string.
   *
   * @param * Given layer of type CReLU.
   * @return The string representation of the layer.
   */
  std::string LayerString(CReLU<>* /*layer*/) const
  {
    return "crelu";
  }

  /**
   * Return the name of the given layer of type Highway as a string.
   *
   * @param * Given layer of type Highway.
   * @return The string representation of the layer.
   */
  std::string LayerString(Highway<>* /*layer*/) const
  {
    return "highway";
  }

  /**
   * Return the name of the given layer of type GRU as a string.
   *
   * @param * Given layer of type GRU.
   * @return The string representation of the layer.
   */
  std::string LayerString(GRU<>* /*layer*/) const
  {
    return "gru";
  }

  /**
   * Return the name of the given layer of type Glimpse as a string.
   *
   * @param * Given layer of type Glimpse.
   * @return The string representation of the layer.
   */
  std::string LayerString(Glimpse<>* /*layer*/) const
  {
    return "glimpse";
  }

  /**
   * Return the name of the given layer of type FastLSTM as a string.
   *
   * @param * Given layer of type FastLSTM.
   * @return The string representation of the layer.
   */
  std::string LayerString(FastLSTM<>* /*layer*/) const
  {
    return "fastlstm";
  }

  /**
   * Return the name of the given layer of type WeightNorm as a string.
   *
   * @param * Given layer of type WeightNorm.
   * @return The string representation of the layer.
   */
  std::string LayerString(WeightNorm<>* /*layer*/) const
  {
    return "weightnorm";
  }

  /**
   * Return the name of the layer of specified type as a string.
   *
   * @param * Given layer of any type.
   * @return A string declaring that the layer is unsupported.
   */
  template<typename T>
  std::string LayerString(T* /*layer*/) const
  {
    return "unsupported";
  }

  //! Overload function call.
  std::string operator()(MoreTypes layer) const
  {
    return layer.apply_visitor(*this);
  }

  //! Overload function call.
  template<typename LayerType>
  std::string operator()(LayerType* layer) const
  {
    return LayerString(layer);
  }
};
