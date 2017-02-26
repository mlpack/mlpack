/**
 * @file add_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Add() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_ADD_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_ADD_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * AddVisitor exposes the Add() method of the given module.
 */
class AddVisitor : public boost::static_visitor<void>
{
 public:
  //! Exposes the Add() method of the given module.
  template<typename T>
  AddVisitor(T newLayer);

  //! Exposes the Add() method.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The layer that should be added.
  LayerTypes newLayer;

  //! Only add the layer if the module implements the Add() function.
  template<typename T>
  typename std::enable_if<
      HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
  LayerAdd(T* layer) const;

  //! Do not add the layer if the module doesn't implement the Add() function.
  template<typename T>
  typename std::enable_if<
      !HasAddCheck<T, void(T::*)(LayerTypes)>::value, void>::type
  LayerAdd(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "add_visitor_impl.hpp"

#endif
