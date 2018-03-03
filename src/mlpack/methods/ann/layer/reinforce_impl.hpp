/**
 * @file reinforce_impl.hpp
 * @author Shangtong Zhang
 *
 * Implementation of the Reinforce class, which implements the policy
 * gradient algorithm .
 */
#ifndef MLPACK_METHODS_ANN_LAYER_REINFORCE_IMPL_HPP
#define MLPACK_METHODS_ANN_LAYER_REINFORCE_IMPL_HPP

// In case it hasn't yet been included.
#include "reinforce.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
template<typename eT>
double Reinforce<InputDataType, OutputDataType>::Forward(
    const arma::Mat<eT>&&, arma::Mat<eT>&& )
{ return 0; }

template<typename InputDataType, typename OutputDataType>
template<typename DataType>
void Reinforce<InputDataType, OutputDataType>::Backward(
    const DataType&& , DataType&& advantage, DataType&& g)
{ g = advantage; }

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void Reinforce<InputDataType, OutputDataType>::Serialize(
    Archive& /* ar */, const unsigned int /* version */)
{ /* Nothing to do here. */ }

} // namespace ann
} // namespace mlpack

#endif
