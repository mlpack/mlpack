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
  // type.  We'll try first by removing any instances of <>.
  const size_t loc = cppType.find("<>");
  if (loc != std::string::npos)
    cppType.replace(loc, 2, "");

  // Let's just replace any invalid characters with valid '_' characters.
  std::replace(cppType.begin(), cppType.end(), '<', '_');
  std::replace(cppType.begin(), cppType.end(), '>', '_');
  std::replace(cppType.begin(), cppType.end(), ' ', '_');
  std::replace(cppType.begin(), cppType.end(), ',', '_');

  return cppType;
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
