/**
 * @file reset_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Reset() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_RESET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ResetVisitor executes the Reset() function.
 */
class ResetVisitor : public boost::static_visitor<void>
{
 public:
  //! Execute the Reset() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! Execute the Reset() function for a module which implements the Reset()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasResetCheck<T, void(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;

  //! Execute the Reset() function for a module which implements the Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasResetCheck<T, void(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;

  //! Execute the Reset() function for a module which implements the Reset()
  //! and Model() function.
  template<typename T>
  typename std::enable_if<
      HasResetCheck<T, void(T::*)()>::value &&
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;

  //! Do not execute the Reset() function for a module which doesn't implement
  // the Reset() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasResetCheck<T, void(T::*)()>::value &&
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  ResetParameter(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "reset_visitor_impl.hpp"

#endif
