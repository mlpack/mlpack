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
  delta(std::move(delta))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void BackwardVisitor::operator()(LayerType* layer) const
{
  layer->Backward(std::move(input), std::move(error), std::move(delta));
}

} // namespace ann
} // namespace mlpack

#endif
