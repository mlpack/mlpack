/**
 * @file util.hpp
 * @author Vasyl Teliman
 *
 * Various utilities used to generate Java bindings
 */
#ifndef MLPACK_BINDINGS_JAVA_UTIL_HPP
#define MLPACK_BINDINGS_JAVA_UTIL_HPP

#include <ios>
#include <streambuf>
#include <string>
#include <vector>
#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Concise way to redirect streams in C++ using RAII
 */
class RedirectStream
{
 public:
  RedirectStream(std::ios& from, const std::ios& to)
  : from(from),
    buffer(from.rdbuf(to.rdbuf()))
  {}

  ~RedirectStream()
  {
    from.rdbuf(buffer);
  }

 private:
  std::ios& from;
  std::streambuf* buffer;
};

/**
 * Generate bindings for model parameters
 */
void PrintModelPointers(const std::vector<util::ParamData>& in, const std::vector<util::ParamData>& out);

/**
 * Convert snake_case name to CamelCase
 */
std::string ToCamelCase(const std::string& s);

} // namespace java
} // namespace bindings
} // namespace mlpack

#endif
