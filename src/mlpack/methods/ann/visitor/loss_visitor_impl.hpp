/**
 * @file methods/ann/visitor/loss_visitor_impl.hpp
 * @author Atharva Khandait
 *
 * Implementation of the Loss() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_LOSS_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_LOSS_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "loss_visitor.hpp"

namespace mlpack {
namespace ann {

//! LossVisitor visitor class.
template<typename LayerType>
inline double LossVisitor::operator()(LayerType* layer) const
{
  return LayerLoss(layer);
}

inline double LossVisitor::operator()(MoreTypes layer) const
{
  return layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    !HasLoss<T, double(T::*)()>::value &&
    !HasModelCheck<T>::value, double>::type
LossVisitor::LayerLoss(T* /* layer */) const
{
  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasLoss<T, double(T::*)()>::value &&
    !HasModelCheck<T>::value, double>::type
LossVisitor::LayerLoss(T* layer) const
{
  return layer->Loss();
}

template<typename T>
inline typename std::enable_if<
    !HasLoss<T, double(T::*)()>::value &&
    HasModelCheck<T>::value, double>::type
LossVisitor::LayerLoss(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    double loss = boost::apply_visitor(LossVisitor(),
        layer->Model()[layer->Model().size() - 1 - i]);

    if (loss != 0)
    {
      return loss;
    }
  }

  return 0;
}

template<typename T>
inline typename std::enable_if<
    HasLoss<T, double(T::*)()>::value &&
    HasModelCheck<T>::value, double>::type
LossVisitor::LayerLoss(T* layer) const
{
  double loss = layer->Loss();

  if (loss == 0)
  {
    for (size_t i = 0; i < layer->Model().size(); ++i)
    {
      loss = boost::apply_visitor(LossVisitor(),
          layer->Model()[layer->Model().size() - 1 - i]);

      if (loss != 0)
      {
        return loss;
      }
    }
  }

  return loss;
}

} // namespace ann
} // namespace mlpack

#endif
