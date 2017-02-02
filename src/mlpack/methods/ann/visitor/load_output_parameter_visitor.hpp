/**
 * @file load_output_parameter_visitor.hpp
 * @author Marcus Edel
 *
 * This file provides an abstraction for the OutputParameter() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_LOAD_OUTPUT_PARAMETER_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_LOAD_OUTPUT_PARAMETER_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * LoadOutputParameterVisitor restores the output parameter using the given
 * parameter set.
 */
class LoadOutputParameterVisitor : public boost::static_visitor<void>
{
 public:
  //! Restore the output parameter given a parameter set.
  LoadOutputParameterVisitor(std::vector<arma::mat>&& parameter);

  //! Restore the output parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! The parameter set.
  std::vector<arma::mat>&& parameter;

  //! Restore the output parameter for a module which doesn't implement the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  OutputParameter(T* layer) const;

  //! Restore the output parameter for a module which implements the Model()
  //! function.
  template<typename T>
  typename std::enable_if<
      HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
  OutputParameter(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "load_output_parameter_visitor_impl.hpp"

#endif
