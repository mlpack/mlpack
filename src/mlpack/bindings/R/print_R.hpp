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
 * @param doc Documentation for the binding.
 * @param functionName Name of the function (i.e. "pca").
 */
void PrintR(const util::BindingDetails& doc,
            const std::string& functionName);

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
