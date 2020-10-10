/**
 * @file bindings/markdown/print_type_doc.hpp
 * @author Ryan Curtin
 *
 * Print documentation for a given type, depending on the current language (as
 * set in BindingInfo).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_TYPE_DOC_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_TYPE_DOC_HPP

#include "binding_info.hpp"

#include <mlpack/bindings/cli/print_type_doc.hpp>
#include <mlpack/bindings/python/print_type_doc.hpp>
#include <mlpack/bindings/julia/print_type_doc.hpp>
#include <mlpack/bindings/go/print_type_doc.hpp>
#include <mlpack/bindings/R/print_type_doc.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Print the type of a parameter into the output string.  The type printed
 * depends on the current setting of BindingInfo::Language().
 */
template<typename T>
std::string PrintTypeDoc(util::ParamData& data)
{
  if (BindingInfo::Language() == "cli")
  {
    return cli::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "python")
  {
    return python::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "julia")
  {
    return julia::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "go")
  {
    return go::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else if (BindingInfo::Language() == "r")
  {
    return r::PrintTypeDoc<typename std::remove_pointer<T>::type>(data);
  }
  else
  {
    throw std::invalid_argument("PrintTypeDoc(): unknown "
        "BindingInfo::Language() " + BindingInfo::Language() + "!");
  }
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

#endif
