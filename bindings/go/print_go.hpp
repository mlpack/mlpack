/**
 * @file bindings/go/print_go.hpp
 * @author Yasmine Dumouchel
 *
 * Given a list of ParamData structures, emit a .go file defining the
 * Go bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_GO_HPP
#define MLPACK_BINDINGS_GO_PRINT_GO_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given a list of parameter definition and program documentation, print a
 * generated .go file to stdout.
 *
 * @param params Instantiated Params struct with program options.
 * @param doc Documentation for the program.
 * @param functionName Name of the function (i.e. "pca").
 * @param bindingName Name of the binding as registered with IO.
 */
void PrintGo(util::Params& params,
             const util::BindingDetails& doc,
             const std::string& functionName,
             const std::string& bindingName);


} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
