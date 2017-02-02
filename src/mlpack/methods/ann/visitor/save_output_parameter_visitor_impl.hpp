/**
 * @file save_output_parameter_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the OutputParameter() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_SAVE_OUTPUT_PARAMETER_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_SAVE_OUTPUT_PARAMETER_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "load_output_parameter_visitor.hpp"

namespace mlpack {
namespace ann {

//! SaveOutputParameterVisitor visitor class.
inline SaveOutputParameterVisitor::SaveOutputParameterVisitor(
    std::vector<arma::mat>&& parameter) : parameter(std::move(parameter))
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void SaveOutputParameterVisitor::operator()(LayerType* layer) const
{
  OutputParameter(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
SaveOutputParameterVisitor::OutputParameter(T* layer) const
{
  parameter.push_back(layer->OutputParameter());
}

template<typename T>
inline typename std::enable_if<
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, void>::type
SaveOutputParameterVisitor::OutputParameter(T* layer) const
{
  parameter.push_back(layer->OutputParameter());

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(SaveOutputParameterVisitor(std::move(parameter)),
        layer->Model()[i]);
  }
}

} // namespace ann
} // namespace mlpack

#endif
