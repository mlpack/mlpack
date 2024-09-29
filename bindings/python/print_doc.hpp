/**
 * @file bindings/python/print_doc.hpp
 * @author Ryan Curtin
 *
 * Print documentation (as part of a docstring) for a Python binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_DOC_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_DOC_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>
#include "get_printable_type.hpp"

namespace mlpack {
namespace bindings {
namespace python {

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
 * @param * (output) Unused parameter.
 */
template<typename T>
void PrintDoc(util::ParamData& d,
              const void* input,
              void* /* output */)
{
  const size_t indent = *((size_t*) input);
  std::ostringstream oss;
  oss << " - ";
  oss << GetValidName(d.name);
  oss << " (";
  oss << GetPrintableType<typename std::remove_pointer<T>::type>(d) << "): "
      << d.desc;

  // Print a default, if possible.
  if (!d.required)
  {
    // Call the correct overload to get the default value directly.
    if (d.cppType == "std::string" || d.cppType == "double" ||
        d.cppType == "int" || d.cppType == "std::vector<int>" ||
        d.cppType == "std::vector<std::string>" ||
        d.cppType == "std::vector<double>")
    {
      std::string defaultValue = DefaultParamImpl<T>(d);
      oss << "  Default value " << defaultValue << ".";
    }
  }

  std::cout << util::HyphenateString(oss.str(), indent + 4);
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
