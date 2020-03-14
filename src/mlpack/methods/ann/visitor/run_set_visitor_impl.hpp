/**
 * @file run_set_visitor_impl.hpp
 * @author Saksham Bansal
 *
 * Implementation of the Run() function layer abstraction.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_IMPL_HPP
#define MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_IMPL_HPP

// In case it hasn't been included yet.
#include "run_set_visitor.hpp"

namespace mlpack {
namespace ann {

//! RunSetVisitor visitor class.
inline RunSetVisitor::RunSetVisitor(
    const bool run) : run(run)
{
  /* Nothing to do here. */
}

template<typename LayerType>
inline void RunSetVisitor::operator()(LayerType* layer) const
{
  LayerRun(layer);
}

inline void RunSetVisitor::operator()(MoreTypes layer) const
{
  layer.apply_visitor(*this);
}

template<typename T>
inline typename std::enable_if<
    HasRunCheck<T, bool&(T::*)(void)>::value &&
    HasModelCheck<T>::value, void>::type
RunSetVisitor::LayerRun(T* layer) const
{
  layer->Run() = run;

  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(RunSetVisitor(run),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    !HasRunCheck<T, bool&(T::*)(void)>::value &&
    HasModelCheck<T>::value, void>::type
RunSetVisitor::LayerRun(T* layer) const
{
  for (size_t i = 0; i < layer->Model().size(); ++i)
  {
    boost::apply_visitor(RunSetVisitor(run),
        layer->Model()[i]);
  }
}

template<typename T>
inline typename std::enable_if<
    HasRunCheck<T, bool&(T::*)(void)>::value &&
    !HasModelCheck<T>::value, void>::type
RunSetVisitor::LayerRun(T* layer) const
{
  layer->Run() = run;
}

template<typename T>
inline typename std::enable_if<
    !HasRunCheck<T, bool&(T::*)(void)>::value &&
    !HasModelCheck<T>::value, void>::type
RunSetVisitor::LayerRun(T* /* input */) const
{
  /* Nothing to do here. */
}

} // namespace ann
} // namespace mlpack

#endif
