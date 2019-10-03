/**
 * @file backward_visitor_impl.hpp
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
inline BackwardVisitor::BackwardVisitor(arma::mat&& input,
                                        arma::mat&& error,
                                        arma::mat&& delta) :
  input(std::move(input)),
  error(std::move(error)),
  delta(std::move(delta)),
  index(0),
  hasIndex(false)
{
  /* Nothing to do here. */
}

inline BackwardVisitor::BackwardVisitor(arma::mat&& input,
                                        arma::mat&& error,
                                        arma::mat&& delta,
                                        const size_t index) :
  input(std::move(input)),
  error(std::move(error)),
  delta(std::move(delta)),
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
  layer->Backward(std::move(input), std::move(error), std::move(delta));
}

template<typename T>
inline typename std::enable_if<
    HasRunCheck<T, bool&(T::*)(void)>::value, void>::type
BackwardVisitor::LayerBackward(T* layer, arma::mat& /* input */) const
{
  if (!hasIndex)
  {
    layer->Backward(std::move(input), std::move(error),
        std::move(delta));
  }
  else
  {
    layer->Backward(std::move(input), std::move(error),
        std::move(delta), index);
  }
}

} // namespace ann
} // namespace mlpack

#endif
