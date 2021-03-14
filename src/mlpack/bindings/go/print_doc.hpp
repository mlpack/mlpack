/**
 * @file bindings/go/print_doc.hpp
 * @author Yashwant Singh
 * @author Yasmine Dumouchel
 *
 * Print documentation for a Go binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_DOC_HPP
#define MLPACK_BINDINGS_GO_PRINT_DOC_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>
#include "get_go_type.hpp"
#include <mlpack/bindings/util/camel_case.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Print the docstring documentation for a given parameter.  You are responsible
 * for setting up the line---this does not handle indentation or anything.  This
 * is meant to produce a line of documentation describing a single parameter.
 *
 * The indent parameter (void* input, which should be a pointer to a size_t)
 * should be passed to know how much to indent for a new line.
 *
 * @param d Parameter data struct.
 * @param input Pointer to size_t containing indent.
 * @param isLower is pointer to bool if isLower is true then parameter is Output
 * or Required Input.
 */
template<typename T>
void PrintDoc(util::ParamData& d,
              const void* input,
              void* isLower)
{
  const size_t indent = *((size_t*) input);
  bool Lower = *((bool*) isLower);
  std::ostringstream oss;
  oss << " - ";
  oss << util::CamelCase(d.name, Lower) << " (";
  oss << GetGoType<typename std::remove_pointer<T>::type>(d) << "): "
      << d.desc;

  // Print a default, if possible.
  if (!d.required)
  {
    if (d.cppType == "std::string")
    {
      oss << "  Default value '" << boost::any_cast<std::string>(d.value)
          << "'.";
    }
    else if (d.cppType == "double")
    {
      oss << "  Default value " << boost::any_cast<double>(d.value) << ".";
    }
    else if (d.cppType == "int")
    {
      oss << "  Default value " << boost::any_cast<int>(d.value) << ".";
    }
  }

  std::cout << util::HyphenateString(oss.str(), indent + 4);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
