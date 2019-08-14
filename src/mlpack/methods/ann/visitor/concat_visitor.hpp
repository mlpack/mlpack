/**
 * @file concat_visitor.hpp
 * @author Saksham Bansal
 *
 * This file provides an abstraction for the Concat() function for different
 * layers.
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
 * ConcatVisitor update the concat matrix using the given matrix.
 */
class ConcatVisitor : public boost::static_visitor<void>
{
 public:
  //! Update the concat given the concat matrix.
  ConcatVisitor(arma::mat&& concat);

  //! Update the concat matrix.
  template<typename LayerType>
  void operator()(LayerType *layer) const;

 private:
  //! The concat matrix.
  arma::mat&& concat;

  //! Do not update the concat set if the module doesn't implement the
  //! Concat() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasConcatCheck<T, P&(T::*)()>::value, void>::type
  LayerConcat(T* layer, P& output) const;

  //! Update the concat set if the module implements the Concat()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      HasConcatCheck<T, P&(T::*)()>::value, void>::type
  LayerConcat(T* layer, P& output) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "concat_visitor_impl.hpp"

#endif
