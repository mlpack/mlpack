/**
 * @filename strip_type.hpp
 * @author Ryan Curtin
 *
 * Given a C++ type name, turn it into something that has no special characters
 * that can simply be printed.
 */
#ifndef MLPACK_BINDINGS_JULIA_STRIP_TYPE_HPP
#define MLPACK_BINDINGS_JULIA_STRIP_TYPE_HPP

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Given a C++ type name, turn it into something that has no special characters
 * that can simply be printed.  This is similar to but not identical to
 * mlpack::bindings::python::StripType().
 *
 * @param cppType C++ type as a string.
 * @return Stripped type with no special characters.
 */
inline std::string StripType(std::string cppType)
{
  // Basically what we need to do is strip any '<' (template bits) from the
  // type.
  if (cppType.find("<") != std::string::npos)
  {
    // Are there any template parameters?  Or is it the default?
    const size_t loc = cppType.find("<>");
    if (loc != std::string::npos)
    {
      // Convert it from "<>".
      cppType.replace(loc, 2, "");
    }
    else
    {
      // Let's just replace the '<' and '>' with a valid '_' character.
      while (cppType.find("<") != std::string::npos)
        cppType.replace(cppType.find("<"), 1, "_");
      while (cppType.find(">") != std::string::npos)
        cppType.replace(cppType.find(">"), 1, "_");
      while (cppType.find(" ") != std::string::npos)
        cppType.replace(cppType.find(" "), 1, "_");
      while (cppType.find(",") != std::string::npos)
        cppType.replace(cppType.find(","), 1, "_");
    }
  }

  return cppType;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
