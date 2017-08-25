/**
 * @file rho_set_visitor_impl.hpp
 * @author Marcus Edel
 *
 * Implementation of the LayerRho() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RHO_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_RHO_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "rho_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! LayerRho visitor class.
inline RhoSetVisitor::RhoSetVisitor(const size_t rho) : rho(rho)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline bool RhoSetVisitor::operator()(LayerType* layer) const
{
  return LayerRho(layer);
}

template<typename T>
inline typename std::enable_if<
    !HasRho<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
RhoSetVisitor::LayerRho(T* /* layer */) const
{
  return false;
}

template<typename T>
inline typename std::enable_if<
    HasRho<T, size_t&(T::*)()>::value &&
    !HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
RhoSetVisitor::LayerRho(T* layer) const
{
  layer->Rho() = rho;
  return true;
}

template<typename T>
inline typename std::enable_if<
    !HasRho<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
RhoSetVisitor::LayerRho(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
    boost::apply_visitor(RhoSetVisitor(rho), layer->Model()[i]);

  return true;
}

template<typename T>
inline typename std::enable_if<
    HasRho<T, size_t&(T::*)()>::value &&
    HasModelCheck<T, std::vector<LayerTypes>&(T::*)()>::value, bool>::type
RhoSetVisitor::LayerRho(T* layer) const
{
  layer->Rho() = rho;

  for (size_t i = 0; i < layer->Model().size(); ++i)
    boost::apply_visitor(RhoSetVisitor(rho), layer->Model()[i]);

  return true;
}

} // namespace ann
} // namespace mlpack

#endif