/**
 * @file bindings/markdown/get_printable_type.hpp
 * @author Ryan Curtin
 *
 * Get the printable type of the parameter.  This depends on
 * BindingInfo::Language() to choose which language to return the type for.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_TYPE_HPP
#define MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_TYPE_HPP

#include "binding_info.hpp"

#include <mlpack/bindings/cli/get_printable_type.hpp>
#include <mlpack/bindings/python/get_printable_type.hpp>
#include <mlpack/bindings/julia/get_printable_type.hpp>
#include <mlpack/bindings/go/get_printable_type.hpp>
#include <mlpack/bindings/R/get_printable_type.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Print the type of a parameter into the output string.  The type printed
 * depends on the current setting of BindingInfo::Language().
 */
template<typename T>
void GetPrintableType(util::ParamData& data,
                      const void* /* input */,
                      void* output)
{
  if (BindingInfo::Language() == "cli")
  {
    *((std::string*) output) =
        cli::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "python")
  {
    *((std::string*) output) =
        python::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "julia")
  {
    *((std::string*) output) =
        julia::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "go")
  {
    *((std::string*) output) =
        go::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "r")
  {
    *((std::string*) output) =
        r::GetPrintableType<typename std::remove_pointer<T>::type>(data);
  }
  else
  {
    throw std::invalid_argument("GetPrintableType(): unknown "
        "BindingInfo::Language(): " + BindingInfo::Language() + "!");
  }
}

/**
 * Print the type of a parameter.  The type printed depends on the current
 * setting of BindingInfo::Language().
 */
template<typename T>
std::string GetPrintableType(util::ParamData& data)
{
  std::string output;
  GetPrintableType<T>(data, (void*) NULL, (void*) &output);
  return output;
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
