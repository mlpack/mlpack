/**
 * @file bindings/markdown/print_param_table.hpp
 * @author Nippun Sharma
 *
 * Prints a table with the parameters of a binding based on certain
 * conditions.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_PARAM_TABLE_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_PARAM_TABLE_HPP

#include <mlpack/prereqs.hpp>

/**
 * Print a table in markdown format that contains
 * a list of parameters.
 * 
 * @param bindingName Parameters corresponding to bindingName.
 * @param language Parameters for a particular language.
 * @param params Params object.
 * @param headers Which headers to print (eg: Name, Default, etc.).
 * @param paramsSet To prevent printing a parameter more than once.
 * @param onlyHyperParams If true, print only hyper-parameters.
 * @param onlyMatrixParams If true, print only matrix-parameters.
 * @param onlyInputParams If true, print only input-parameters.
 * @param onlyOutputParams If true, print only output-parameters.
 */
void PrintParamTable(const std::string& bindingName,
                     mlpack::util::Params& params,
                     const std::set<std::string>& headers,
                     std::unordered_set<std::string>& paramsSet,
                     const bool onlyHyperParams,
                     const bool onlyMatrixParams,
                     const bool onlyInputParams,
                     const bool onlyOutputParams);

#endif
