/**
 * @file forward_with_memory_visitor.hpp
 * @author Sumedh Ghaisas
 *
 * This file provides an abstraction for the ForwardWithMemory() function which
 * also accepts the current memory conten for different layers and automatically
 * directs any parameter to the right layer type. This visitor is useful the
 * networks which deals with external memory content.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_FORWARD_WITH_MEMORY_TEST_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_FORWARD_WITH_MEMORY_TEST_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * ForwardWithMemoryVisitor executes the ForwardWithMemory() function given the
 * input, memory and output parameter.
 */
class ForwardWithMemoryTestVisitor : public boost::static_visitor<void>
{
 public:
  /**
   * Execute the FowardWithMemory() function given the input, current memory
   * and output parameter.
   */
  ForwardWithMemoryTestVisitor(arma::mat&& input,
                               arma::mat&& memory,
                               arma::mat&& output);

  //! Execute the FowardWithMemory() function.
  template<typename LayerType>
  void operator()(LayerType* layer) const;

 private:
  //! Execute the ForwardWithMemory() function with memory for a module which
  //! implements ForwardWithMemory() function
  template<typename T>
  typename std::enable_if<
      HasForwardWithMemoryTestCheck<T, void(T::*)(arma::mat&&,
      const arma::mat&&, arma::mat&&)>::value, void>::type
  ForwardWithMemoryTest(T* layer) const;

  //! Do not execute the ForwardWithMemory() function for a module which
  //! doesn't implement ForwardWithMemory() function.
  template<typename T>
  typename std::enable_if<
      !HasForwardWithMemoryTestCheck<T, void(T::*)(arma::mat&&,
      const arma::mat&&, arma::mat&&)>::value, void>::type
  ForwardWithMemoryTest(T* layer) const;

  //! The input parameter set.
  arma::mat&& input;

  //! The memory content.
  arma::mat&& memory;

  //! The output parameter set.
  arma::mat&& output;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "forward_with_memory_test_visitor_impl.hpp"

#endif
