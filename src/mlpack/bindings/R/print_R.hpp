/**
 * @file bindings/R/print_R.hpp
 * @author Yashwant Singh Parihar
 *
 * Definition of utility PrintR() function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_R_HPP
#define MLPACK_BINDINGS_R_PRINT_R_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Print the code for a .R binding for an mlpack program to stdout.
 *
 * @param params Instantiated Params object for this binding.
 * @param functionName Name of the function (i.e. "pca").
 * @param bindingName Name of the binding (as specified by BINDING_NAME).
 */
void PrintR(util::Params& params,
            const std::string& functionName,
            const std::string& bindingName);

/**
 * Split the last part of a method.
 *
 * This assumes '_' as the delimiter.
 *
 * @param bindingname Binding name as e.g. 'linear_regression_train'.
 */
inline std::string SplitBindingName(const std::string& s)
{
  std::vector<std::string> tokens;
  std::string token;
  std::stringstream ss(s);
  const char delimiter = '_';

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  std::string out = "";
  size_t n = tokens.size();
  for (size_t i = 0; i < n - 1; i++) {
    out += tokens[i];
    if (i < n - 2)
      out += delimiter;
  }

  return out;
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
