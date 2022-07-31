/**
 * @file bindings/julia/print_doc.hpp
 * @author Ryan Curtin
 *
 * Print inline documentation for a single option.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_DOC_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_DOC_HPP

namespace mlpack {
namespace bindings {
namespace julia {

template<typename T>
void PrintDoc(util::ParamData& d, const void* /* input */, void* output)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  std::ostringstream& oss = *((std::ostringstream*) output);

  oss << "`" << juliaName << "::" << GetJuliaType<typename std::remove_pointer
      <T>::type>(d) << "`: " << d.desc;

  // Print a default, if possible.  Defaults aren't printed for matrix or model
  // parameters.
  if (!d.required)
  {
    if (d.cppType == "std::string" ||
        d.cppType == "double" ||
        d.cppType == "int" ||
        d.cppType == "bool")
    {
      oss << "  Default value `";
      if (d.cppType == "std::string")
      {
        oss << MLPACK_ANY_CAST<std::string>(d.value);
      }
      else if (d.cppType == "double")
      {
        oss << MLPACK_ANY_CAST<double>(d.value);
      }
      else if (d.cppType == "int")
      {
        oss << MLPACK_ANY_CAST<int>(d.value);
      }
      else if (d.cppType == "bool")
      {
        oss << (MLPACK_ANY_CAST<bool>(d.value) ? "true" : "false");
      }
      oss << "`." << std::endl;
    }
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
