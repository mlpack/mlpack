/**
 * @file add_merge_impl.hpp
 * @author Marcus Edel
 *
 * Definition of the AddMerge module which accumulates the output of the given
 * modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP

// In case it hasn't yet been included.
#include "add_merge_impl.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
AddMerge<InputDataType, OutputDataType>::AddMerge()
{
  // Nothing to do here.
}

template <typename InputDataType, typename OutputDataType>
template<typename InputType, typename OutputType>
void AddMerge<InputDataType, OutputDataType>::Forward(
    const InputType&& /* input */, OutputType&& output)
{
  output = boost::apply_visitor(outputParameterVisitor, network.front());

  for (size_t i = 1; i < network.size(); ++i)
  {
    output += boost::apply_visitor(outputParameterVisitor, network[i]);
  }
}

template<typename InputDataType, typename OutputDataType>
template<typename eT>
void AddMerge<InputDataType, OutputDataType>::Backward(
    const arma::Mat<eT>&& /* input */, arma::Mat<eT>&& gy, arma::Mat<eT>&& g)
{
  g = gy;
}


template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void AddMerge<InputDataType, OutputDataType>::Serialize(
    Archive& ar, const unsigned int /* version */)
{
  ar & data::CreateNVP(network, "network");
}

} // namespace ann
} // namespace mlpack

#endif
