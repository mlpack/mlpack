/**
 * @file layer_names.hpp
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

class LayerNameVisitor : public boost::static_visitor<std::string>
{
 public:
  LayerNameVisitor()
  {
  }

  std::string LayerString(AtrousConvolution<>* /*layer*/) const
  {
    return "atrousconvolution";
  }

  std::string LayerString(AlphaDropout<>* /*layer*/) const
  {
    return "alphadropout";
  }

  std::string LayerString(BatchNorm<>* /*layer*/) const
  {
    return "batchnorm";
  }

  std::string LayerString(Constant<>* /*layer*/) const
  {
    return "constant";
  }

  std::string LayerString(Convolution<>* /*layer*/) const
  {
    return "convolution";
  }

  std::string LayerString(DropConnect<>* /*layer*/) const
  {
    return "dropconnect";
  }

  std::string LayerString(Dropout<>* /*layer*/) const
  {
    return "dropout";
  }

  std::string LayerString(FlexibleReLU<>* /*layer*/) const
  {
    return "frelu";
  }

  std::string LayerString(LayerNorm<>* /*layer*/) const
  {
    return "layernorm";
  }

  std::string LayerString(Linear<>* /*layer*/) const
  {
    return "linear";
  }

  std::string LayerString(LinearNoBias<>* /*layer*/) const
  {
    return "linearnobias";
  }

  std::string LayerString(MaxPooling<>* /*layer*/) const
  {
    return "maxpooling";
  }

  std::string LayerString(MeanPooling<>* /*layer*/) const
  {
    return "meanpooling";
  }

  std::string LayerString(MultiplyConstant<>* /*layer*/) const
  {
    return "multiplyconstant";
  }

  std::string LayerString(ReLULayer<>* /*layer*/) const
  {
    return "relu";
  }

  std::string LayerString(TransposedConvolution<>* /*layer*/) const
  {
    return "transposedconvolution";
  }

  std::string LayerString(IdentityLayer<>* /*layer*/) const
  {
    return "identity";
  }

  std::string LayerString(TanHLayer<>* /*layer*/) const
  {
    return "tanh";
  }

  std::string LayerString(ELU<>* /*layer*/) const
  {
    return "elu";
  }

  std::string LayerString(HardTanH<>* /*layer*/) const
  {
    return "hardtanh";
  }

  std::string LayerString(LeakyReLU<>* /*layer*/) const
  {
    return "leakyrelu";
  }

  std::string LayerString(PReLU<>* /*layer*/) const
  {
    return "prelu";
  }

  std::string LayerString(SigmoidLayer<>* /*layer*/) const
  {
    return "sigmoid";
  }

  std::string LayerString(LogSoftMax<>* /*layer*/) const
  {
    return "logsoftmax";
  }

  template<typename T>
  std::string LayerString(T* /*layer*/) const
  {
    return "unsupported";
  }

  template<typename LayerType>
  std::string operator()(LayerType* layer) const
  {
    return LayerString(layer);
  }
};
