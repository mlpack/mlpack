/**
 * @file name_visitor.hpp
 * @author Prince Gupta
 *
 * This file provides an abstraction for the Name() function for
 * different layers and automatically provides the corresponding name.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_VISITOR_NAME_VISITOR_HPP
#define MLPACK_METHODS_ANN_VISITOR_NAME_VISITOR_HPP

#include <mlpack/methods/ann/layer/layer_traits.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>

#include <boost/variant.hpp>

namespace mlpack {
namespace ann {

/**
 * NameVisitor exposes the name of the given module.
 */
class NameVisitor : public boost::static_visitor<std::string>
{
 public:
  //! Return the layer name.
  template<typename LayerType>
  std::string operator()(LayerType* layer) const;

  std::string operator()(MoreTypes layer) const;

 private:
  //! Return "unknown" if the module doesn't implement the Name()
  //! function.
  template<typename T>
  typename std::enable_if<
      !HasName<T, std::string&(T::*)()>::value, std::string>::type
  LayerName(T* layer) const;

  //! Return the name if the module implements the Name() funtion.
  template<typename T>
  typename std::enable_if<
      HasName<T, std::string&(T::*)()>::value, std::string>::type
  LayerName(T* layer) const;
};

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "name_visitor_impl.hpp"

#endif
