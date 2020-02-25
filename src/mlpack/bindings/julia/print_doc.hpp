/**
 * @file print_doc.hpp
 * @author Ryan Curtin
 *
 * Print inline documentation for a single option.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_DOC_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_DOC_HPP

namespace mlpack {
namespace bindings {
namespace julia {

template<typename T>
void PrintDoc(const util::ParamData& d, const void* /* input */, void* output)
{
  // "type" is a reserved keyword or function.
  const std::string juliaName = (d.name == "type") ? "type_" : d.name;

  std::ostringstream& oss = *((std::ostringstream*) output);

  oss << "`" << juliaName << "::" << GetJuliaType<T>() << "`: " << d.desc;

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
        oss << boost::any_cast<std::string>(d.value);
      }
      else if (d.cppType == "double")
      {
        oss << boost::any_cast<double>(d.value);
      }
      else if (d.cppType == "int")
      {
        oss << boost::any_cast<int>(d.value);
      }
      else if (d.cppType == "bool")
      {
        oss << (boost::any_cast<bool>(d.value) ? "true" : "false");
      }
      oss << "`." << std::endl;
    }
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif

