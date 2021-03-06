/**
 * @file methods/ann/visitor/backward_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the Backward() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_BACKWARD_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "backward_visitor.hpp"

namespace mlpack {
namespace ann {

//! BackwardVisitor visitor class.
inline BackwardVisitor::BackwardVisitor(const arma::mat& input,
                                        const arma::mat& error,
                                        arma::mat& delta) :
  input(input),
  error(error),
  delta(delta),
  index(0),
  hasIndex(false)
{
  /* Nothing to do here. */
}

inline BackwardVisitor::BackwardVisitor(const arma::mat& input,
                                        const arma::mat& error,
                                        arma::mat& delta,
                                        const size_t index) :
  input(input),
  error(error),
  delta(delta),
  index(index),
  hasIndex(true)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void BackwardVisitor::operator()(LayerType* layer) const
{
  LayerBackward(layer, layer->OutputParameter());
}

inline void BackwardVisitor::operator()(MoreTypes layer) const
{
  layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    !HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
BackwardVisitor::LayerBackward(T* layer, arma::mat& /* input */) const
{
  layer->Backward(input, error, delta);
}

template<typename T>
inline typename std::enable_if<
    HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
BackwardVisitor::LayerBackward(T* layer, arma::mat& /* input */) const
{
  if (!hasIndex)
  {
    layer->Backward(input, error, delta);
  }
  else
  {
    layer->Backward(input, error, delta, index);
  }
}

} // namespace ann
} // namespace mlpack

#endif
