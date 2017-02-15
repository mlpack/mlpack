/**
 * @file parameters_visitor.hpp
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
#ifndef MLPACK_METHODS_ANN_VISITOR_PARAMETERS_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_PARAMETERS_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ParametersVisitor exposes the parameters set of the given module and stores
 * the parameters set into the given matrix.
 */
class ParametersVisitor : public boost::static_visitor<void>
{
 public:
  //! Store the parameters set into the given parameters matrix.
  ParametersVisitor(arma::mat&& parameters);

  //! Set the parameters set.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The parameters set.
  arma::mat&& parameters;

  //! Do not set the parameters set if the module doesn't implement the
  //! Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      !HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;

  //! Set the parameters set if the module implements the Parameters() function.
  template<typename T, typename P>
  typename std::enable_if<
      HasParametersCheck<T, P&(T::*)()>::value, void>::type
  LayerParameters(T* layer, P& output) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "parameters_visitor_impl.hpp"

#endif
