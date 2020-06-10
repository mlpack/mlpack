/**
 * @file bindings/go/camel_case.hpp
 * @author Yashwant Singh
 *
 * Given a C++ typename that may have template parameters, return stripped and
 * printable versions to be used in Go bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_CAMEL_CASE_HPP
#define MLPACK_BINDINGS_GO_CAMEL_CASE_HPP

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given an snake_case like, e.g., "logistic_regression", return
 * CamelCase(e.g. "LogisticRegression") that can be used in Go code.
 *
 * @param s input string.
 * @param lower is of bool type. If lower is true then output must be
 *     lowerCamelCase else UpperCamelCase.
 */
inline std::string CamelCase(const std::string& s, bool lower)
{
  if (!lower)
    s[0] = std::toupper(s[0]);
  else
    s[0] = std::tolower(s[0]);
  size_t n = s.length();
  size_t resInd = 0;
  for (size_t i = 0; i < n; i++)
  {
    // Check for spaces in the sentence.
    if (s[i] == '_')
    {
      // Conversion into upper case.
      s[i + 1] = toupper(s[i + 1]);
      continue;
    }
    // If not space, copy character.
    else
      s[resInd++] = s[i];
  }
  // Return string to main.
  return s.substr(0, resInd);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
