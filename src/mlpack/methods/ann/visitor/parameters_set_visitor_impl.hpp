/**
 * @file parameters_set_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Parameters() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_PARAMETERS_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_PARAMETERS_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "parameters_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! ParametersSetVisitor visitor class.
inline ParametersSetVisitor::ParametersSetVisitor(arma::mat&& parameters) :
    parameters(std::move(parameters))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ParametersSetVisitor::operator()(LayerType *layer) const
{
  LayerParameters(layer, layer->OutputParameter());
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasParametersCheck<T, P&(T::*)()>::value, void>::type
ParametersSetVisitor::LayerParameters(T* /* layer */, P& /* output */) const
{
  /* Nothing to do here. */
}

template<typename T, typename P>
inline typename std::enable_if<
    HasParametersCheck<T, P&(T::*)()>::value, void>::type
ParametersSetVisitor::LayerParameters(T* layer, P& /* output */) const
{
  layer->Parameters() = parameters;
}

} // namespace ann
} // namespace mlpack

#endif
