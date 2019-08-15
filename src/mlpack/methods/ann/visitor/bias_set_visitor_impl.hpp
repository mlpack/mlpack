/**
 * @file bias_set_visitor_impl.hpp
 * @author Toshal Agrawal
 *
 * Implementation of the Bias() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_BIAS_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_BIAS_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "bias_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! BiasSetVisitor visitor class.
inline BiasSetVisitor::BiasSetVisitor(arma::mat&& weight, const size_t offset) :
    weight(std::move(weight)),
    offset(offset)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline size_t BiasSetVisitor::operator()(LayerType* layer) const
{
  return LayerSize(layer);
}

inline size_t BiasSetVisitor::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    !HasBiasCheck<T, arma::mat&(T::*)()>::value &&
    !HasModelCheck<T>::value, size_t>::type
BiasSetVisitor::LayerSize(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    !HasBiasCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T>::value, size_t>::type
BiasSetVisitor::LayerSize(T* layer) const
{
  size_t modelOffset = 0;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(BiasSetVisitor(
        std::move(weight), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

template<typename T>
inline typename std::enable_if<
    HasBiasCheck<T, arma::mat&(T::*)()>::value &&
    !HasModelCheck<T>::value, size_t>::type
BiasSetVisitor::LayerSize(T* layer) const
{
  layer->Bias() = arma::mat(weight.memptr() + offset,
      layer->Bias().n_rows, layer->Bias().n_cols, false, false);

  return layer->Bias().n_elem;
}

template<typename T>
inline typename std::enable_if<
    HasBiasCheck<T, arma::mat&(T::*)()>::value &&
    HasModelCheck<T>::value, size_t>::type
BiasSetVisitor::LayerSize(T* layer) const
{
  layer->Bias() = arma::mat(weight.memptr() + offset,
      layer->Bias().n_rows, layer->Bias().n_cols, false, false);

  size_t modelOffset = layer->Bias().n_elem;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    modelOffset += boost::apply_visitor(BiasSetVisitor(
        std::move(weight), modelOffset + offset), layer->Model()[i]);
  }

  return modelOffset;
}

} // namespace ann
} // namespace mlpack

#endif
