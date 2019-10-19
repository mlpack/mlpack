/**
 * @file run_set_visitor.hpp
 * @author Saksham Bansal
 *
 * This file provides an abstraction for the Run() function for
 * different layers and automatically directs any parameter to the right layer
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_RUN_SET_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * RunSetVisitor set the run parameter given the
 * run value.
 */
class RunSetVisitor : public boost::static_visitor<void>
{
 public:
  //! Set the run parameter given the current run value.
  RunSetVisitor(const bool run = true);

  //! Set the run parameter.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

  void operator()(MoreTypes layer) const;

 private:
  //! The run parameter.
  const bool run;

  //! Set the run parameter if the module implements the
  //! Run() and Model() function.
  template<typename T>
  typename std::enable_if<
      HasRunCheck<T, bool&(T::*)(void)>::value &&
      HasModelCheck<T>::value, void>::type
  LayerRun(T* layer) const;

  //! Set the run parameter if the module implements the
  //! Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRunCheck<T, bool&(T::*)(void)>::value &&
      HasModelCheck<T>::value, void>::type
  LayerRun(T* layer) const;

  //! Set the run parameter if the module implements the
  //! Run() function.
  template<typename T>
  typename std::enable_if<
      HasRunCheck<T, bool&(T::*)(void)>::value &&
      !HasModelCheck<T>::value, void>::type
  LayerRun(T* layer) const;

  //! Do not set the run parameter if the module doesn't implement the
  //! Run() or Model() function.
  template<typename T>
  typename std::enable_if<
      !HasRunCheck<T, bool&(T::*)(void)>::value &&
      !HasModelCheck<T>::value, void>::type
  LayerRun(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "run_set_visitor_impl.hpp"

#endif
