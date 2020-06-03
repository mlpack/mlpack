/**
 * @file methods/ann/layer/dropconnect_impl.hpp
 * @author Palash Ahuja
 * @author Marcus Edel
 *
 * Implementation of the DropConnect class, which implements a regularizer
 * that randomly sets connections to zero. Preventing units from co-adapting.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_DROPCONNECT_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_DROPCONNECT_IMPL_HPP

// In case it hasn't yet been included.
#include "dropconnect.hpp"

#include "../visitor/delete_visitor.hpp"
#include "../visitor/forward_visitor.hpp"
#include "../visitor/backward_visitor.hpp"
#include "../visitor/gradient_visitor.hpp"
#include "../visitor/parameters_set_visitor.hpp"
#include "../visitor/parameters_visitor.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
DropConnect<InputDataType, OutputDataType>::DropConnect() :
    ratio(0.5),
    scale(2.0),
    deterministic(true)
{
  // Nothing to do here.
}

template<typename InputDataType, typename OutputDataType>
DropConnect<InputDataType, OutputDataType>::DropConnect(
    const size_t inSize,
    const size_t outSize,
    const double ratio) :
    ratio(ratio),
    scale(1.0 / (1 - ratio)),
    baseLayer(new Linear<InputDataType, OutputDataType>(inSize, outSize))
{
  network.push_back(baseLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void DropConnect<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>& input,
    arma::Mat<eT>& output)
{
  // The DropConnect mask will not be multiplied in the deterministic mode
  // (during testing).
  if (deterministic)
  {
    boost::apply_visitor(ForwardVisitor(input, output), baseLayer);
  }
  else
  {
    // Save weights for denoising.
    boost::apply_visitor(ParametersVisitor(denoise), baseLayer);

    // Scale with input / (1 - ratio) and set values to zero with
    // probability ratio.
    mask = arma::randu<arma::Mat<eT> >(denoise.n_rows, denoise.n_cols);
    mask.transform([&](double val) { return (val > ratio); });

    arma::mat tmp = denoise % mask;
    boost::apply_visitor(ParametersSetVisitor(tmp), baseLayer);

    boost::apply_visitor(ForwardVisitor(input, output), baseLayer);

    output = output * scale;
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void DropConnect<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& gy,
    arma::Mat<eT>& g)
{
  boost::apply_visitor(BackwardVisitor(input, gy, g), baseLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void DropConnect<InputDataType, OutputDataType>::Gradient(
    const arma::Mat<eT>& input,
    const arma::Mat<eT>& error,
    arma::Mat<eT>& /* gradient */)
{
  boost::apply_visitor(GradientVisitor(input, error),
      baseLayer);

  // Denoise the weights.
  boost::apply_visitor(ParametersSetVisitor(denoise), baseLayer);
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void DropConnect<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  // Delete the old network first, if needed.
  if (Archive::is_loading::value)
  {
    boost::apply_visitor(DeleteVisitor(), baseLayer);
  }

  ar & BOOST_SERIALIZATION_NVP(ratio);
  ar & BOOST_SERIALIZATION_NVP(scale);
  ar & BOOST_SERIALIZATION_NVP(baseLayer);

  if (Archive::is_loading::value)
  {
    network.clear();
    network.push_back(baseLayer);
  }
}

}  // namespace ann
}  // namespace mlpack

#endif
