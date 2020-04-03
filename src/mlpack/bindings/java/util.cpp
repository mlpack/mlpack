/**
 * @file util.hpp
 * @author Vasyl Teliman
 *
 * Implementation of various utilities used to generate Java bindings
 */

#include <unordered_set>
#include <mlpack/core/util/cli.hpp>
#include <cctype>
#include "util.hpp"

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Generate bindings for model parameters
 */
void PrintModelPointers(const std::vector<util::ParamData>& in, const std::vector<util::ParamData>& out)
{
  std::unordered_set<std::string> types;

  for (const util::ParamData& param : in)
  {
    if (types.count(param.cppType) == 0) {
      types.insert(param.cppType);
      CLI::GetSingleton().functionMap[param.tname]["PrintParamDefn"](param, nullptr, nullptr);
    }
  }

  for (const util::ParamData& param : out)
  {
    if (types.count(param.cppType) == 0) {
      types.insert(param.cppType);
      CLI::GetSingleton().functionMap[param.tname]["PrintParamDefn"](param, nullptr, nullptr);
    }
  }
}

/**
 * Convert snake_case name to CamelCase
 */
std::string ToCamelCase(const std::string& s)
{
  if (s.empty() || s == "_") return "";

  std::string result;
  result += toupper(s[0]);

  for (int i = 1, n = s.size(); i < n; ++i)
  {
    if (s[i] == '_') continue;
    if (s[i - 1] == '_') result += toupper(s[i]);
    else result += s[i];
  }

  return result;
}

} // namespace java
} // namespace bindings
} // namespace mlpack
