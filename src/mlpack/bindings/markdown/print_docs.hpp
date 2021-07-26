/**
 * @file bindings/markdown/print_docs.hpp
 * @author Ryan Curtin
 *
 * Functions to generate Markdown for documentation of bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_DOCS_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_DOCS_HPP

#include <mlpack/prereqs.hpp>

/**
 * Given the current settings of IO, print the header (which will be the
 * navigation tab) for the binding types that are registered for these options.
 *
 * Output is printed to stdout.
 */
void PrintHeaders(const std::string& bindingName,
                  const std::vector<std::string>& languages,
                  const std::vector<bool>& addWrapperDocs);

/**
 * Given the current settings of IO, print Markdown documentation for the
 * binding types that are registered for these options.
 *
 * Output is printed to stdout.
 *
 * @param bindingName The binding name (${BINDING} from CMake).
 * @param languages The set of languages to print documentation for.
 */
void PrintDocs(const std::string& bindingName,
               const std::vector<std::string>& languages,
               const std::vector<std::string>& validMethods,
               const std::vector<bool>& addWrapperDocs);

#endif
