/**
 * @file assign_visitor.hpp
 * @author Shangtong Zhang
 *
 * This file provides an abstraction for assignment between layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_ASSIGN_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_ASSIGN_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * This visitor is to support assignment constructor and
 * assignment operator for neural network module.
 * We want a layer-wise assignment rather than simply duplicate the pointer.
 */
class AssignVisitor : public boost::static_visitor<void>
{
 public:
  template <typename LayerType>
  void operator () (LayerType*, LayerType*) const;

  template <typename LayerType1, typename LayerType2>
  void operator () (LayerType1*, LayerType2*) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation
#include "assign_visitor_impl.hpp"
#endif

