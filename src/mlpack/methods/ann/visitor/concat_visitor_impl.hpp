/**
 * @file concat_visitor_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the Concat() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_CONCAT_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_CONCAT_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "concat_visitor.hpp"

namespace mlpack {
namespace ann {

//! ConcatVisitor visitor class.
inline ConcatVisitor::ConcatVisitor(arma::mat&& concat) :
    concat(std::move(concat))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void ConcatVisitor::operator()(LayerType *layer) const
{
  LayerConcat(layer, layer->OutputParameter());
}

template<typename T, typename P>
inline typename std::enable_if<
    !HasConcatCheck<T, P&(T::*)()>::value, void>::type
ConcatVisitor::LayerConcat(T* /* layer */, P& /* output */) const
{
  /* Nothing to do here. */
}

template<typename T, typename P>
inline typename std::enable_if<
    HasConcatCheck<T, P&(T::*)()>::value, void>::type
ConcatVisitor::LayerConcat(T* layer, P& /* output */) const
{
  layer->Concat() = concat;
}

} // namespace ann
} // namespace mlpack

#endif
