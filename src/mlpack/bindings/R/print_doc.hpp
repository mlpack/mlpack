/**
 * @file bindings/R/print_doc.hpp
 * @author Yashwant Singh Parihar.
 *
 * Print documentation for a R binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_DOC_HPP
#define MLPACK_BINDINGS_R_PRINT_DOC_HPP

#include "get_r_type.hpp"
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Print the docstring documentation for a given parameter.  You are responsible
 * for setting up the line---this does not handle indentation or anything.  This
 * is meant to produce a line of documentation describing a single parameter.
 *
 * The indent parameter (void* input, which should be a pointer to a size_t)
 * should be passed to know how much to indent for a new line.
 *
 * @param d Parameter data struct.
 * @param * (input) Pointer to size_t containing indent.
 * @param output Unused parameter.
 */
template<typename T>
void PrintDoc(util::ParamData& d,
              const void* /* input */,
              void* output)
{
  bool out = *((bool*) output);
  std::ostringstream oss;
  if (out)
    oss << "#' \\item{" << d.name << "}{";
  else
    oss << "#' @param " << d.name << " ";
  oss << d.desc.substr(0, d.desc.size() - 1);
  // Print a default, if possible.
  if (!d.required)
  {
    if (d.cppType == "std::string" ||
        d.cppType == "double" ||
        d.cppType == "int" ||
        d.cppType == "bool")
    {
      oss << ".  Default value \"";
      if (d.cppType == "std::string")
      {
        oss << std::any_cast<std::string>(d.value);
      }
      else if (d.cppType == "double")
      {
        oss << std::any_cast<double>(d.value);
      }
      else if (d.cppType == "int")
      {
        oss << std::any_cast<int>(d.value);
      }
      else if (d.cppType == "bool")
      {
        // If the option is `verbose`, be sure to print the use of the global
        // mlpack package option as a default.
        if (d.name == "verbose")
        {
          oss << "getOption(\"mlpack.verbose\", FALSE)";
        }
        else
        {
          oss << (std::any_cast<bool>(d.value) ? "TRUE" : "FALSE");
        }
      }
      oss << "\"";
    }
  }

  oss << " (" << GetRType<std::remove_pointer_t<T>>(d) << ").";

  if (out)
    oss << "}";

  MLPACK_COUT_STREAM << util::HyphenateString(oss.str(), "#'   ");
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
