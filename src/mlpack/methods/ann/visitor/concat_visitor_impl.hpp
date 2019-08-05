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
inline ConcatVisitor::ConcatVisitor(arma::mat&& input) :
    input(std::move(input))
{
  /* Nothing to do here. */
}

//! ConcatVisitor visitor class.
template<typename LayerType>
inline void ConcatVisitor::operator()(LayerType* layer) const
{
  Concat(layer);
}

template<typename T>
inline typename std::enable_if<
    HasConcatCheck<T, void(T::*)(const size_t)>::value, void>::type
ConcatVisitor::Concat(T* layer) const
{
  layer->Concat() = input;
}

template<typename T>
inline typename std::enable_if<
    !HasConcatCheck<T, void(T::*)(const size_t)>::value, void>::type
ConcatVisitor::Concat(T* /* layer */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
