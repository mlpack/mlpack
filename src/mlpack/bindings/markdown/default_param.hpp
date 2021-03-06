/**
 * @file bindings/markdown/default_param.hpp
 * @author Ryan Curtin
 *
 * Get the default value of the parameter.  This depends on
 * BindingInfo::Language() to choose which language to return the type for.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_BINDINGS_MARKDOWN_DEFAULT_PARAM_HPP
#define MLPACK_BINDINGS_MARKDOWN_DEFAULT_PARAM_HPP

#include "binding_info.hpp"

#include <mlpack/bindings/cli/default_param.hpp>
#include <mlpack/bindings/python/default_param.hpp>
#include <mlpack/bindings/julia/default_param.hpp>
#include <mlpack/bindings/go/default_param.hpp>
#include <mlpack/bindings/R/default_param.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Print the default value of a parameter into the output string.  The type
 * printed depends on the current setting of BindingInfo::Language().
 */
template<typename T>
void DefaultParam(util::ParamData& data,
                  const void* /* input */,
                  void* output)
{
  if (BindingInfo::Language() == "cli")
  {
    *((std::string*) output) =
        cli::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "python")
  {
    *((std::string*) output) =
        python::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "julia")
  {
    *((std::string*) output) =
        julia::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "go")
  {
    *((std::string*) output) =
        go::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "r")
  {
    *((std::string*) output) =
        r::DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
  }
  else
  {
    throw std::invalid_argument("DefaultParam(): unknown "
        "BindingInfo::Language() " + BindingInfo::Language() + "!");
  }
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
