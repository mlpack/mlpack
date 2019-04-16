/**
 * @file camel_case.hpp
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
 */
inline std::string CamelCase(std::string s)
{
  s[0] = std::toupper(s[0]);
  size_t n = s.length();
  size_t res_ind = 0;
  for (size_t i = 0; i < n; i++)
  {
    // check for spaces in the sentence
    if (s[i] == '_')
    {
      // conversion into upper case
      s[i + 1] = toupper(s[i + 1]);
      continue;
    }
    // If not space, copy character
    else
      s[res_ind++] = s[i];
  }
  // return string to main
  return s.substr(0, res_ind);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
