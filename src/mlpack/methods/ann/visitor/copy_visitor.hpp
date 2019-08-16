/**
 * @file copy_visitor.hpp
 * @author Shangtong Zhang
 *
 * This file provides an abstraction for copy between layers.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_COPY_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * This visitor is to support copy constructor for neural network module.
 * We want a layer-wise copy rather than simple duplicate the pointer.
 */
template <typename... CustomLayers>
class CopyVisitor : public boost::static_visitor<LayerTypes<CustomLayers...> >
{
 public:
  template <typename LayerType>
  LayerTypes<CustomLayers...> operator()(LayerType*) const;

  LayerTypes<CustomLayers...> operator()(MoreTypes) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation
#include "copy_visitor_impl.hpp"
#endif

