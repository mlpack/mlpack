/**
 * @file concat_visitor.hpp
 * @author Saksham Bansal
 *
 * Boost static visitor abstraction for calling Concat function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_CONCAT_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_CONCAT_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ConcatVisitor executes the Concat() function.
 */
class ConcatVisitor : public boost::static_visitor<void>
{
 public:
  //! Concat the the given input matrix.
  ConcatVisitor(arma::mat&& input);

  //! Execute the Concat() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  arma::mat&& input;

  //! Execute the Concat() function for a module which implements
  //! the Concat() function.
  template<typename T>
  typename std::enable_if<
      HasConcatCheck<T, void(T::*)(const size_t)>::value, void>::type
  Concat(T* layer) const;

  //! Do not execute the Concat() function for a module which doesn't implement
  // the Concat() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasConcatCheck<T, void(T::*)(const size_t)>::value, void>::type
  Concat(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_visitor_impl.hpp"

#endif
