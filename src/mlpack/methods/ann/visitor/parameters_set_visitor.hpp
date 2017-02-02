/**
 * @file parameters_set_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the Parameters() function for different
 * layers and automatically directs any parameter to the right layer type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_PARAMETERS_SET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_PARAMETERS_SET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ParametersSetVisitor update the parameters set using the given matrix.
 */
class ParametersSetVisitor : public boost::static_visitor<void>
{
 public:
  //! Update the parameters set given the parameters matrix.
  ParametersSetVisitor(arma::mat&& parameters);

  //! Update the parameters set.
  template<typename LayerType>
  void operator()(LayerType *layer) const;

 private:
  //! The parameters set.
  arma::mat&& parameters;

  //! Do not update the parameters set if the module doesn't implement the
  //! Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;

  //! Update the parameters set if the module implements the Parameters()
  //! function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "parameters_set_visitor_impl.hpp"

#endif
